from src.stage_b.fitted_timing import fit_prefill_linear


def test_fit_prefill_linear_perfect_line() -> None:
    # y = 0.01 + 0.0001 * x
    xs = [100, 200, 400, 800]
    ys = [0.01 + 0.0001 * x for x in xs]
    a, b = fit_prefill_linear(xs, ys)
    assert abs(a - 0.01) < 1e-6
    assert abs(b - 0.0001) < 1e-6
