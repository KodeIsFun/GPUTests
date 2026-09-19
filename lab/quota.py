from __future__ import annotations

from dataclasses import dataclass
import re

GPU_JOB_MAX_HOURS = 8
GPU_WEEK_HARD_STOP_HOURS = 24
PROXY_DAY_MAX_USD = 8.0
PROXY_MONTH_MAX_USD = 90.0
PROXY_DAY_TARGET_USD = 3.0
KAGGLE_PROXY_MONTH_TOTAL_USD = 100.0

_HOURS = re.compile(r"GPU\s+(\d+(?:\.\d+)?)h\s+(\d+(?:\.\d+)?)h\s+(\d+(?:\.\d+)?)h")


class BudgetError(ValueError):
    pass


@dataclass(frozen=True)
class GpuQuota:
    used_hours: float
    remaining_hours: float
    total_hours: float


@dataclass(frozen=True)
class ProxyQuota:
    daily_remaining: float
    monthly_remaining: float


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str


def parse_gpu_quota_table(table: str) -> GpuQuota:
    match = _HOURS.search(table)
    if match is None:
        raise BudgetError("could not parse GPU row from kaggle quota table")
    used, remaining, total = (float(g) for g in match.groups())
    return GpuQuota(used_hours=used, remaining_hours=remaining, total_hours=total)


def decide_gpu_job(quota: GpuQuota, budget_hours: float) -> BudgetDecision:
    if budget_hours <= 0:
        return BudgetDecision(allowed=False, reason="budget_hours must be > 0")
    if budget_hours > GPU_JOB_MAX_HOURS:
        return BudgetDecision(
            allowed=False,
            reason=f"job budget_hours {budget_hours} exceeds {GPU_JOB_MAX_HOURS}h session cap",
        )
    if quota.used_hours + budget_hours > GPU_WEEK_HARD_STOP_HOURS:
        return BudgetDecision(
            allowed=False,
            reason=(
                f"job would take weekly used GPU hours "
                f"{quota.used_hours + budget_hours} past hard stop "
                f"{GPU_WEEK_HARD_STOP_HOURS}"
            ),
        )
    if budget_hours > quota.remaining_hours:
        return BudgetDecision(
            allowed=False,
            reason=f"job needs {budget_hours}h but only {quota.remaining_hours}h remain",
        )
    return BudgetDecision(allowed=True, reason="ok")


def decide_proxy_run(
    quota: ProxyQuota,
    estimated_usd: float,
    publish_day: bool = False,
) -> BudgetDecision:
    if estimated_usd <= 0:
        return BudgetDecision(allowed=False, reason="estimated_usd must be > 0")
    daily_cap = PROXY_DAY_MAX_USD if publish_day else PROXY_DAY_TARGET_USD
    if estimated_usd > daily_cap:
        return BudgetDecision(
            allowed=False,
            reason=f"estimated ${estimated_usd} exceeds daily cap ${daily_cap}",
        )
    if estimated_usd > quota.daily_remaining:
        return BudgetDecision(
            allowed=False,
            reason=f"estimated ${estimated_usd} exceeds daily remaining ${quota.daily_remaining}",
        )
    monthly_used = KAGGLE_PROXY_MONTH_TOTAL_USD - quota.monthly_remaining
    if monthly_used + estimated_usd > PROXY_MONTH_MAX_USD:
        return BudgetDecision(
            allowed=False,
            reason=(
                f"estimated ${estimated_usd} would take monthly used "
                f"${monthly_used + estimated_usd} past ${PROXY_MONTH_MAX_USD}"
            ),
        )
    if estimated_usd > quota.monthly_remaining:
        return BudgetDecision(
            allowed=False,
            reason=f"estimated ${estimated_usd} exceeds monthly remaining ${quota.monthly_remaining}",
        )
    return BudgetDecision(allowed=True, reason="ok")


def require_gpu_job(quota: GpuQuota, budget_hours: float) -> None:
    decision = decide_gpu_job(quota, budget_hours)
    if not decision.allowed:
        raise BudgetError(decision.reason)


def require_proxy_run(
    quota: ProxyQuota, estimated_usd: float, publish_day: bool = False
) -> None:
    decision = decide_proxy_run(quota, estimated_usd, publish_day=publish_day)
    if not decision.allowed:
        raise BudgetError(decision.reason)
