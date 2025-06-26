//! Rolling walk-forward training / prediction on CPU XGBoost.

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, s};
use xgboost_rs::{
    Booster, DMatrix,
    parameters::{
        BoosterParametersBuilder, BoosterType, TrainingParametersBuilder,
        learning::{LearningTaskParametersBuilder, Objective},
        tree::{TreeBoosterParametersBuilder, TreeMethod},
    },
};

/// Trains and predicts in 24-hour hops until the end of the data set.
pub fn walk_forward_predict(
    x: &Array2<f32>,
    y: &[f32],
    lookback: usize,
    horizon: usize,
    num_boost_round: u32,
    verbose: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.nrows(), y.len(), "X / y length mismatch");

    let mut preds = Vec::<f32>::new();
    let mut actuals = Vec::<f32>::new();

    let end = x.nrows();
    let steps = (end - lookback) / horizon;
    let pb = ProgressBar::new(steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} {wide_bar} {pos}/{len} · {elapsed_precise} ETA {eta_precise}",
        )
        .unwrap(),
    );
    pb.set_message("Walk-forward");

    let mut start = lookback;
    let mut step = 0usize;

    while start + horizon <= end {
        // TRAIN ------------------------------------------------------------------
        let train_x = x.slice(s![start - lookback..start, ..]).to_owned();
        let train_y = &y[start - lookback..start];

        let train_flat = train_x.into_raw_vec(); // row-major contiguous
        let mut dtrain = DMatrix::from_dense(&train_flat, lookback).unwrap();
        dtrain.set_labels(train_y).unwrap();

        // MODEL PARAMS -----------------------------------------------------------
        let learning = LearningTaskParametersBuilder::default()
            .objective(Objective::RegLinear) // plain squared-error  :contentReference[oaicite:1]{index=1}
            .build()
            .unwrap();

        let tree = TreeBoosterParametersBuilder::default()
            .max_depth(6)
            .eta(0.05)
            .subsample(0.8)
            .colsample_bytree(0.8)
            .tree_method(TreeMethod::Hist) // CPU histogram router  :contentReference[oaicite:2]{index=2}
            .build()
            .unwrap();

        let booster_params = BoosterParametersBuilder::default()
            .booster_type(BoosterType::Tree(tree))
            .learning_params(learning)
            .build()
            .unwrap();

        let training_params = TrainingParametersBuilder::default()
            .dtrain(&dtrain)
            .booster_params(booster_params)
            .boost_rounds(num_boost_round)
            .build()
            .unwrap();

        let booster = Booster::train(&training_params).unwrap();

        // PREDICT ---------------------------------------------------------------
        let test_x = x.slice(s![start..start + horizon, ..]).to_owned();
        let test_flat = test_x.into_raw_vec();
        let dtest = DMatrix::from_dense(&test_flat, horizon).unwrap();

        let y_pred = booster.predict(&dtest).unwrap();

        preds.extend_from_slice(&y_pred);
        actuals.extend_from_slice(&y[start..start + horizon]);

        if verbose {
            let rmse = rmse(&y_pred, &actuals[actuals.len() - horizon..]);
            println!(
                "[step {:03}] rows {}-{} : RMSE={:.4}",
                step,
                start,
                start + horizon - 1,
                rmse
            );
        }

        step += 1;
        start += horizon;
        pb.inc(1);
    }

    pb.finish_with_message("done");
    (preds, actuals)
}

#[inline]
fn rmse(pred: &[f32], truth: &[f32]) -> f32 {
    let mse: f32 = pred
        .iter()
        .zip(truth)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / pred.len() as f32;
    mse.sqrt()
}
