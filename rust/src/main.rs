//! CPU walk-forward benchmark entry-point.

use std::time::Instant;

use ndarray::Array2;
use xgboost_benchmark::{generate_data::create_dataset, walkforward::walk_forward_predict};

mod generate_data;
mod walkforward;

fn main() {
    // 0) generate data ----------------------------------------------------------
    let t0 = Instant::now();
    let (x, y) = create_dataset(
        None, // default 455 days × 24
        200, 24, 42, 0.10, true,
    );

    // 1) walk-forward -----------------------------------------------------------
    let (preds, actuals) =
        walk_forward_predict(&x, &y, 365 * 24, 24, 200, /* verbose = */ true);

    // 2) overall RMSE -----------------------------------------------------------
    let overall_rmse = {
        let rmse = preds
            .iter()
            .zip(actuals.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / preds.len() as f32;
        rmse.sqrt()
    };
    println!("\n=== overall RMSE: {:.4} ===", overall_rmse);

    println!("Total wall-clock: {:.2?}", t0.elapsed());
}
