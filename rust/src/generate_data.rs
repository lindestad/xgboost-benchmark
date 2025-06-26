//! src/generate_data.rs
//!
//! Generate a synthetic hourly data set and append 24 lags per feature.
//!
//! • 1 year + 3 months  →  455 days × 24 h  = 10 920 rows  
//! • 200 base features; target depends on ~10 % of them  
//! • (24 lags + current) × 200  = 5 000 columns  
//! • Drop first 24 rows so every sample has full lag history.
//!
//! Returns `(X, y)` where  
//! - `X`  = `Array2<f32>` shape (10 896, 5 000)  
//! - `y`  = `Vec<f32>`   length 10 896
use ndarray::{Array1, Array2, Axis, Zip, s};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Synthesize the data set and add lagged columns.
///
/// * `hours` — override total length (default = 455 days × 24)  
/// * `n_features` — raw feature count  
/// * `n_lags` — maximum lag to add (inclusive)  
/// * `random_state` — RNG seed for reproducibility  
/// * `fraction_relevant` — share of features that influence *y*  
/// * `include_current` — keep t-0 features alongside lags
pub fn create_dataset(
    hours: Option<usize>,
    n_features: usize,
    n_lags: usize,
    random_state: u64,
    fraction_relevant: f32,
    include_current: bool,
) -> (Array2<f32>, Vec<f32>) {
    // ─────────────────────────────────────────────────────────── 1. RAW FEATURES
    let hours = hours.unwrap_or((365 + 90) * 24);
    let mut rng = rand::rngs::StdRng::seed_from_u64(random_state);
    let x_raw: Array2<f32> = Array2::random_using((hours, n_features), StandardNormal, &mut rng);

    // ─────────────────────────────────────────────────────────── 2. TARGET VECTOR
    //   linear mix of a random 10 % subset  +  daily seasonality  +  noise
    let n_relevant = ((fraction_relevant * n_features as f32).round() as usize).max(1);

    let mut idx: Vec<usize> = (0..n_features).collect();
    idx.shuffle(&mut rng);
    let relevant = &idx[..n_relevant];

    let beta: Vec<f32> = (0..n_relevant).map(|_| rng.gen_range(-1.0..=1.0)).collect();

    // linear component
    let mut y = x_raw
        .columns()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| relevant.contains(i))
        .zip(beta.iter())
        .fold(Array1::<f32>::zeros(hours), |acc, ((_, col), w)| {
            acc + col.to_owned() * *w
        });

    // seasonality + iid noise
    Zip::from(&mut y)
        .and(&Array1::from_iter(0..hours))
        .for_each(|yi, &t| {
            let season = (2.0 * std::f32::consts::PI * ((t % 24) as f32) / 24.0).sin();
            *yi += 0.3 * season + rng.sample::<f32, _>(StandardNormal) * 0.1;
        });

    // ─────────────────────────────────────────────────────────── 3. LAGGED BLOCKS
    let start_lag = if include_current { 0 } else { 1 };
    let mut lag_blocks = Vec::with_capacity(n_lags + 1 - start_lag);

    for k in start_lag..=n_lags {
        // slice rows [n_lags-k .. hours-k)
        let view = x_raw.slice(s![n_lags - k..hours - k, ..]).to_owned();
        lag_blocks.push(view);
    }

    // (hours - n_lags) × (|lags|·n_features)
    let x_lagged = ndarray::concatenate(Axis(1), &lag_blocks).unwrap();
    let y = y.slice(s![n_lags..]).to_vec();

    (x_lagged, y)
}

// ───────────────────────────────────────────────────────────── Smoke-test
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shapes() {
        let (x, y) = create_dataset(None, 200, 24, 42, 0.10, true);
        assert_eq!(x.nrows(), 10_896);
        assert_eq!(x.ncols(), 5_000);
        assert_eq!(y.len(), 10_896);
    }
}
