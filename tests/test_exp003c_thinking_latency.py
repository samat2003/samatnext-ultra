from scripts.exp003c_thinking_latency import latency_stats, percentile


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0


def test_latency_stats_computes_cv():
    stats = latency_stats([10.0, 20.0, 30.0])
    assert stats.mean_us == 20.0
    assert stats.p50_us == 20.0
    assert stats.cv > 0.0
