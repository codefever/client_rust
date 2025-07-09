//! Module implementing an Open Metrics summary.
//!
//! See [`Summary`] for details.

use std::sync::{Arc};

use parking_lot::RwLock;
use quantiles::ckms::CKMS;

use crate::{
    encoding::{EncodeMetric, MetricEncoder, NoLabelSet},
    metrics::{MetricType, TypedMetric},
};

/// Open Metrics [`Summary`] to measure distributions of discrete events.
///
/// A Summary captures individual observations from an event or sample stream and
/// summarizes them in a manner similar to traditional summary statistics:
/// 1. sum of observations
/// 2. observation count
/// 3. rank estimations (quantiles) over a sliding time window
///
/// The Summary maintains multiple time-windowed streams to provide quantile
/// estimates over a configurable time period. It uses the CKMS (Cormode-Keller-
/// Muthuswamy-Salihoglu) algorithm for efficient quantile estimation.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// // Create a summary that tracks 50th, 90th, and 99th percentiles
/// // over a 5-minute window with 5 buckets and 1% error tolerance
/// let summary = Summary::new(
///     Duration::from_secs(300), // 5 minutes
///     5,                        // 5 buckets
///     vec![0.5, 0.9, 0.99],    // 50th, 90th, 99th percentiles
///     0.01,                     // 1% error tolerance
/// );
///
/// // Record some observations
/// summary.observe(1.0);
/// summary.observe(2.5);
/// summary.observe(3.7);
///
/// // Get current summary statistics
/// let (sum, count, quantiles) = summary.get();
/// println!("Sum: {}, Count: {}", sum, count);
/// for (quantile, value) in quantiles {
///     println!("{}th percentile: {}", quantile * 100.0, value);
/// }
/// ```
#[derive(Clone)]
pub struct Summary(Arc<SummaryMetricsImpl>);

impl std::fmt::Debug for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let this = self.0.as_ref();

        f.debug_struct("SummaryMetrics")
            .field("stream_duration", &this.stream_duration)
            .field("max_age_buckets", &this.max_age_buckets)
            .field("target_error", &this.target_error)
            .field("quantile_values", &self.get())
            .finish()
    }
}

struct SummaryMetricsImpl {
    inner: RwLock<InnerSummary>,
    stream_duration: std::time::Duration,
    max_age_buckets: usize,
    target_quantile: Vec<f64>,
    target_error: f64,
}

struct InnerSummary {
    sum: f64,
    count: u64,
    quantile_streams: Vec<CKMS<f64>>,
    // head_stream is like a cursor which carries the index
    // of the stream in the quantile_streams that we want to query.
    head_stream_idx: usize,
    // timestamp at which the head_stream_idx was last rotated.
    last_rotated_timestamp: quanta::Instant,
}

impl Summary {
    /// Create a new [`Summary`].
    pub fn new(
        max_age: std::time::Duration,
        max_age_buckets: usize,
        target_quantile: Vec<f64>,
        target_error: f64,
    ) -> Self {
        if target_quantile.iter().any(|&x| x > 1.0 || x < 0.0) {
            panic!("target_quantile out of range");
        }
        if max_age_buckets == 0 {
            panic!("max_age_buckets must be greater than 0");
        }

        Self(Arc::new(SummaryMetricsImpl {
            inner: RwLock::new(InnerSummary {
                sum: 0.0,
                count: 0,
                quantile_streams: vec![CKMS::new(target_error); max_age_buckets],
                head_stream_idx: 0,
                last_rotated_timestamp: quanta::Instant::now(),
            }),
            stream_duration: max_age / max_age_buckets as u32,
            max_age_buckets,
            target_quantile,
            target_error,
        }))
    }

    /// Observe the given value.
    pub fn observe(&self, value: impl Into<f64>) {
        let mut inner = self.0.inner.write();
        self.maybe_rotate_streams(&mut inner);
        let value = value.into();

        inner.sum += value;
        inner.count += 1;

        // insert quantiles into all streams/buckets.
        for quantile in &mut inner.quantile_streams {
            quantile.insert(value);
        }
    }

    /// Retrieve the values of the summary metric.
    pub fn get(&self) -> (f64, u64, Vec<(f64, f64)>) {
        let this = self.0.as_ref();
        let mut inner = this.inner.write();
        self.maybe_rotate_streams(&mut inner);
        drop(inner);

        let inner = this.inner.read();
        let sum = inner.sum;
        let count = inner.count;
        let head_stream = &inner.quantile_streams[inner.head_stream_idx];
        let mut quantile_values = Vec::with_capacity(this.target_quantile.len());

        for quantile in this.target_quantile.iter() {
            if let Some(value) = head_stream.query(*quantile) {
                quantile_values.push((*quantile, value.1));
            }
        }

        (sum, count, quantile_values)
    }

    fn maybe_rotate_streams(&self, inner: &mut InnerSummary) {
        let this = self.0.as_ref();
        let mut reset_count = 0;
        let now = quanta::Instant::now();

        while now >= inner.last_rotated_timestamp + this.stream_duration {
            inner.last_rotated_timestamp += this.stream_duration;

            // Reset current buckets
            inner.quantile_streams[inner.head_stream_idx] = CKMS::new(this.target_error);
            // Advance
            inner.head_stream_idx += 1;
            if inner.head_stream_idx == this.max_age_buckets {
                inner.head_stream_idx = 0;
            }

            reset_count += 1;
            if reset_count >= this.max_age_buckets {
                // No more buckets to reset, stop rotating
                inner.last_rotated_timestamp = now;
                break;
            }
        }
    }
}

impl TypedMetric for Summary {
    const TYPE: MetricType = MetricType::Summary;
}

impl EncodeMetric for Summary {
    fn encode(&self, mut encoder: MetricEncoder) -> Result<(), std::fmt::Error> {
        let (sum, count, quantiles) = self.get();
        encoder.encode_summary::<NoLabelSet>(sum, count, &quantiles)
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
            0.01,
        );
        summary.observe(1.0);
        summary.observe(5.0);
        summary.observe(10.0);

        let (s, c, q) = summary.get();
        assert_eq!(16.0, s);
        assert_eq!(3, c);
        assert_eq!(vec![(0.5, 5.0), (0.9, 10.0), (0.99, 10.0)], q);
    }

    #[test]
    #[should_panic(expected = "target_quantile out of range")]
    fn summary_panic() {
        Summary::new(
            std::time::Duration::from_secs(5),
            10,
            vec![1.0, 5.0, 9.0],
            0.01,
        );
    }
}
