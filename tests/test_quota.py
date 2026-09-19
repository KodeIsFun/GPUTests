from lab.quota import (
    GPU_JOB_MAX_HOURS,
    GPU_WEEK_HARD_STOP_HOURS,
    PROXY_DAY_MAX_USD,
    PROXY_MONTH_MAX_USD,
    BudgetDecision,
    GpuQuota,
    ProxyQuota,
    decide_gpu_job,
    decide_proxy_run,
    parse_gpu_quota_table,
)


def test_parse_live_shaped_gpu_table() -> None:
    table = """
resource  used   remaining  total   refreshAt
--------  -----  ---------  ------  -------------------
GPU       0.00h  30.00h     30.00h  2026-09-26T00:00:00
TPU       0.00h  20.00h     20.00h  2026-09-26T00:00:00
"""
    quota = parse_gpu_quota_table(table)
    assert quota == GpuQuota(used_hours=0.0, remaining_hours=30.0, total_hours=30.0)


def test_parse_partially_used_gpu() -> None:
    table = """
resource  used    remaining  total
GPU       12.50h  17.50h     30.00h
TPU       0.00h   20.00h     20.00h
"""
    quota = parse_gpu_quota_table(table)
    assert quota.used_hours == 12.5
    assert quota.remaining_hours == 17.5


def test_refuse_job_over_eight_hours() -> None:
    quota = GpuQuota(used_hours=0.0, remaining_hours=30.0, total_hours=30.0)
    decision = decide_gpu_job(quota, budget_hours=10)
    assert decision.allowed is False
    assert "8" in decision.reason


def test_refuse_when_week_hard_stop_would_be_crossed() -> None:
    quota = GpuQuota(used_hours=22.0, remaining_hours=8.0, total_hours=30.0)
    decision = decide_gpu_job(quota, budget_hours=4)
    assert decision.allowed is False
    assert str(GPU_WEEK_HARD_STOP_HOURS) in decision.reason


def test_allow_job_under_hard_stop() -> None:
    quota = GpuQuota(used_hours=2.0, remaining_hours=28.0, total_hours=30.0)
    decision = decide_gpu_job(quota, budget_hours=3)
    assert decision == BudgetDecision(allowed=True, reason="ok")


def test_refuse_proxy_over_daily_target_on_normal_day() -> None:
    proxy = ProxyQuota(daily_remaining=10.0, monthly_remaining=100.0)
    decision = decide_proxy_run(proxy, estimated_usd=4.0)
    assert decision.allowed is False
    assert "3.0" in decision.reason


def test_publish_day_may_use_eight() -> None:
    proxy = ProxyQuota(daily_remaining=10.0, monthly_remaining=100.0)
    decision = decide_proxy_run(proxy, estimated_usd=8.0, publish_day=True)
    assert decision.allowed is True


def test_refuse_proxy_over_daily_max() -> None:
    proxy = ProxyQuota(daily_remaining=10.0, monthly_remaining=100.0)
    decision = decide_proxy_run(proxy, estimated_usd=8.01, publish_day=True)
    assert decision.allowed is False
    assert str(PROXY_DAY_MAX_USD) in decision.reason


def test_refuse_proxy_when_daily_remaining_too_low() -> None:
    proxy = ProxyQuota(daily_remaining=1.5, monthly_remaining=90.0)
    decision = decide_proxy_run(proxy, estimated_usd=2.0)
    assert decision.allowed is False
    assert "daily" in decision.reason.lower()


def test_refuse_proxy_when_monthly_cap_would_be_crossed() -> None:
    proxy = ProxyQuota(daily_remaining=8.0, monthly_remaining=15.0)
    # 100 - 15 = 85 already used; 85 + 6 = 91 > 90 month usable
    decision = decide_proxy_run(proxy, estimated_usd=6.0, publish_day=True)
    assert decision.allowed is False
    assert str(PROXY_MONTH_MAX_USD) in decision.reason


def test_allow_cheap_proxy_run() -> None:
    proxy = ProxyQuota(daily_remaining=10.0, monthly_remaining=100.0)
    decision = decide_proxy_run(proxy, estimated_usd=2.0)
    assert decision.allowed is True


def test_constants_match_plan() -> None:
    assert GPU_JOB_MAX_HOURS == 8
    assert GPU_WEEK_HARD_STOP_HOURS == 24
    assert PROXY_DAY_MAX_USD == 8.0
    assert PROXY_MONTH_MAX_USD == 90.0
