//! Module implementing an Open Metrics summary.
//!
//! See [`Summary`] for details.

use std::sync::Arc;

use sketches_ddsketch::{Config, DDSketch};
use parking_lot::RwLock;

use crate::{
    encoding::{EncodeMetric, MetricEncoder},
    metrics::{MetricType, TypedMetric},
};

/// Open Metrics [`Summary`] to measure distributions of discrete events.
///
/// Quantiles are computed over a sliding time window using DDSketch, which
/// provides relative-error guarantees (0.01% by default) for arbitrary f64
/// values including negatives. Each sub-bucket covers
/// `max_age / max_age_buckets`; observations older than `max_age` are dropped
/// from quantile calculations while `sum` and `count` remain cumulative.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use prometheus_client::metrics::summary::Summary;
///
/// let summary = Summary::new(
///     Duration::from_secs(300),
///     5,
///     vec![0.5, 0.9, 0.99],
/// );
/// summary.observe(0.042); // e.g. 42 ms latency in seconds
/// ```
#[derive(Clone)]
pub struct Summary(Arc<SummaryImpl>);

impl std::fmt::Debug for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Summary")
            .field("bucket_duration", &self.0.bucket_duration)
            .field("max_buckets", &self.0.max_buckets)
            .field("quantile_values", &self.get())
            .finish()
    }
}

fn new_sketch() -> DDSketch {
    // alpha=0.001 (0.01% relative error), max_bins=32768, min_value=1ns
    DDSketch::new(Config::new(0.0001, 32_768, 1.0e-9))
}

struct Bucket {
    begin: quanta::Instant,
    sketch: DDSketch,
}

struct Inner {
    /// Cumulative sum of all observations (not windowed).
    sum: f64,
    /// Cumulative count of all observations (not windowed).
    count: u64,
    /// Time-windowed buckets, ordered newest-first.
    buckets: Vec<Bucket>,
}

struct SummaryImpl {
    inner: RwLock<Inner>,
    bucket_duration: std::time::Duration,
    max_bucket_duration: std::time::Duration,
    max_buckets: usize,
    target_quantile: Vec<f64>,
}

impl Summary {
    /// Create a new [`Summary`].
    ///
    /// The sliding window has total duration `max_age`, divided into
    /// `max_age_buckets` sub-buckets. Quantile precision uses DDSketch
    /// defaults (0.01% relative error).
    pub fn new(
        max_age: std::time::Duration,
        max_age_buckets: usize,
        target_quantile: Vec<f64>,
    ) -> Self {
        if target_quantile.iter().any(|&x| x > 1.0 || x < 0.0) {
            panic!("target_quantile out of range");
        }
        if max_age_buckets == 0 {
            panic!("max_age_buckets must be greater than 0");
        }
        let bucket_duration = max_age / max_age_buckets as u32;
        if bucket_duration.is_zero() {
            panic!("max_age too small for max_age_buckets: bucket_duration would be zero");
        }

        Self(Arc::new(SummaryImpl {
            inner: RwLock::new(Inner { sum: 0.0, count: 0, buckets: Vec::new() }),
            bucket_duration,
            max_bucket_duration: max_age,
            max_buckets: max_age_buckets,
            target_quantile,
        }))
    }

    /// Observe the given value.
    ///
    /// Non-finite values (NaN, ±infinity) are silently ignored.
    pub fn observe(&self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let now = quanta::Instant::now();
        let mut inner = self.0.inner.write();
        inner.sum += value;
        inner.count += 1;
        self.record_into_buckets(&mut inner, value, now);
    }

    /// Retrieve current (sum, count, quantiles).
    ///
    /// `sum` and `count` are cumulative. Quantiles reflect only the active
    /// sliding window.
    pub fn get(&self) -> (f64, u64, Vec<(f64, f64)>) {
        let now = quanta::Instant::now();
        let inner = self.0.inner.read();
        let cutoff = now.checked_sub(self.0.max_bucket_duration);

        // Merge all non-expired buckets into a single sketch.
        let mut merged = new_sketch();
        inner
            .buckets
            .iter()
            .filter(|b| cutoff.map_or(true, |c| b.begin > c))
            .for_each(|b| {
                merged.merge(&b.sketch).expect("sketches must be compatible");
            });

        let quantile_values = self.0
            .target_quantile
            .iter()
            .map(|&q| {
                let v = if merged.count() == 0 {
                    f64::NAN
                } else {
                    merged.quantile(q).ok().flatten().unwrap_or(f64::NAN)
                };
                (q, v)
            })
            .collect();

        (inner.sum, inner.count, quantile_values)
    }

    fn record_into_buckets(&self, inner: &mut Inner, value: f64, now: quanta::Instant) {
        let this = &self.0;

        // Buckets are newest-first. Walk until we find one that contains `now`
        // or determine that `now` is newer than all existing buckets.
        for bucket in &mut inner.buckets {
            let end = bucket.begin + this.bucket_duration;
            if now > end {
                // `now` is newer than this bucket — need a new head bucket.
                break;
            }
            if now >= bucket.begin {
                bucket.sketch.add(value);
                return;
            }
        }

        // Purge expired buckets before potentially inserting a new one.
        if let Some(cutoff) = now.checked_sub(this.max_bucket_duration) {
            inner.buckets.retain(|b| b.begin > cutoff);
        }

        // If `now` predates all remaining buckets, drop the observation from
        // the window (sum/count are already updated).
        if !inner.buckets.is_empty() && now <= inner.buckets.last().unwrap().begin {
            return;
        }

        let mut sketch = new_sketch();
        sketch.add(value);

        if inner.buckets.is_empty() {
            inner.buckets.push(Bucket { begin: now, sketch });
            return;
        }

        // Align new bucket to the grid defined by the current head bucket.
        let reftime = inner.buckets[0].begin;
        let mut begin = reftime + this.bucket_duration;
        let mut end = begin + this.bucket_duration;
        while now < begin || now >= end {
            begin += this.bucket_duration;
            end += this.bucket_duration;
        }
        inner.buckets.truncate(this.max_buckets - 1);
        inner.buckets.insert(0, Bucket { begin, sketch });
    }
}

impl TypedMetric for Summary {
    const TYPE: MetricType = MetricType::Summary;
}

impl EncodeMetric for Summary {
    fn encode(&self, mut encoder: MetricEncoder) -> Result<(), std::fmt::Error> {
        let (sum, count, quantiles) = self.get();
        encoder.encode_summary(sum, count, &quantiles)
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary() {
        let summary = Summary::new(
            std::time::Duration::from_secs(5),
            10,
            vec![0.5, 0.9, 0.99],
        );
        // DDSketch uses rank = floor(q * (n-1)), so small n produces degenerate
        // results.  Use 100 observations for well-separated quantile values.
        for i in 1..=100 {
            summary.observe(i as f64);
        }

        let (s, c, q) = summary.get();
        assert_eq!(5050.0, s);
        assert_eq!(100, c);
        assert_eq!(3, q.len());

        // p50 → rank 49 → ~50, p90 → rank 89 → ~90, p99 → rank 98 → ~99
        // DDSketch relative error is 0.1%, so ±1 is very generous.
        assert!((q[0].1 - 50.0).abs() < 1.0, "p50: {}", q[0].1);
        assert!((q[1].1 - 90.0).abs() < 1.0, "p90: {}", q[1].1);
        assert!((q[2].1 - 99.0).abs() < 1.0, "p99: {}", q[2].1);
    }

    #[test]
    #[should_panic(expected = "target_quantile out of range")]
    fn summary_panic_quantile_out_of_range() {
        Summary::new(std::time::Duration::from_secs(5), 10, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "bucket_duration would be zero")]
    fn summary_panic_zero_bucket_duration() {
        // 1ns / 2 = 0ns via integer division → must panic
        Summary::new(std::time::Duration::from_nanos(1), 2, vec![0.5]);
    }

    #[test]
    fn empty_window_quantiles_are_nan() {
        let summary = Summary::new(
            std::time::Duration::from_secs(5),
            10,
            vec![0.5, 0.9],
        );
        let (sum, count, q) = summary.get();
        assert_eq!(0.0, sum);
        assert_eq!(0, count);
        assert_eq!(2, q.len());
        assert!(q[0].1.is_nan(), "p50 should be NaN when empty");
        assert!(q[1].1.is_nan(), "p90 should be NaN when empty");
    }

    #[test]
    fn observe_non_finite_is_ignored() {
        let summary = Summary::new(
            std::time::Duration::from_secs(5),
            10,
            vec![0.5],
        );
        summary.observe(1.0);
        summary.observe(f64::NAN);
        summary.observe(f64::INFINITY);
        summary.observe(f64::NEG_INFINITY);
        summary.observe(2.0);

        let (sum, count, _) = summary.get();
        assert_eq!(3.0, sum);
        assert_eq!(2, count);
    }
}
