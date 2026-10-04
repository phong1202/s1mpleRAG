"""Shared token bucket, kept in Redis.

Why Redis rather than counting inside each worker: the quota belongs to the
whole system. Twenty workers each counting "I have made ten calls" total two
hundred, and none of them can tell.

Why Lua: "is there a token left" and "take one" have to be a single
uninterruptible step. As two commands, two workers both read one token
remaining and both spend it.
"""

import random
from functools import lru_cache

import redis as redis_lib

from app.config import get_settings
from app.exceptions import AppException, ErrorCode

# Idle buckets are dropped after an hour. Any bucket in use refreshes this on
# every call, so only abandoned keys expire.
_IDLE_TTL_MS = 3_600_000


class RateLimited(Exception):
    """The bucket said "wait", not "no". Raised instead of calling Celery's
    retry() from inside the work itself: retry() raises Celery's Retry, an
    Exception subclass, which a stage's generic `except Exception` swallowed
    -- recording a failed attempt, then publishing a second retry, so each
    deferral doubled the task. Only the stage task turns this into a retry.

    `partial` carries whatever the deferred run already paid for, so the
    stage can save it and the retry resume rather than start over."""

    def __init__(self, countdown: float, partial: list | None = None) -> None:
        super().__init__(f"rate limited, retry in {countdown:.2f}s")
        self.countdown = countdown
        self.partial = partial


# KEYS[1] = bucket key; ARGV = capacity, refill_per_sec, tokens
#
# The clock is read inside the script with TIME rather than passed in. Taking
# it in the caller costs a second round trip, and leaves a gap in which the
# timestamp no longer matches the state being read. Redis 7 allows TIME in
# scripts because it replicates effects rather than commands.
_SCRIPT = """
local key       = KEYS[1]
local capacity  = tonumber(ARGV[1])
local refill    = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])

local clock  = redis.call('TIME')
local now_ms = tonumber(clock[1]) * 1000 + tonumber(clock[2]) / 1000

if requested > capacity then
  return {0, -1}
end

local state  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now_ms
end

-- max(0, ...) so a clock that steps backwards cannot drain the bucket.
tokens = math.min(capacity, tokens + math.max(0, now_ms - ts) / 1000 * refill)

local allowed = 0
local wait_ms = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
elseif refill <= 0 then
  -- Nothing will ever be added, so this is unsatisfiable in the same sense
  -- as asking for more than the bucket holds.
  wait_ms = -1
else
  wait_ms = math.ceil((requested - tokens) / refill * 1000)
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, ARGV[4])
return {allowed, wait_ms}
"""


class TokenBucket:
    def __init__(self, key: str, capacity: int, refill_per_sec: float) -> None:
        self._redis = redis_lib.from_url(get_settings().redis_url)
        self._script = self._redis.register_script(_SCRIPT)
        self.key = key
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec

    def acquire(self, tokens: int = 1) -> tuple[bool, int]:
        """Return (allowed, wait_ms).

        wait_ms is -1 when the request can never succeed -- larger than the
        bucket, or a bucket that does not refill -- so callers can fail fast
        instead of retrying forever.
        """
        allowed, wait_ms = self._script(
            keys=[self.key],
            args=[self.capacity, self.refill_per_sec, tokens, _IDLE_TTL_MS],
        )
        return bool(allowed), int(wait_ms)


@lru_cache
def get_bucket(name: str) -> TokenBucket:
    """Four buckets, because requests and tokens are separate quotas, and chat
    and embeddings are separate quotas again. Sharing one would let either
    starve the other in a way the provider never asked for."""
    settings = get_settings()
    per_minute = {
        "chat_rpm": settings.rl_chat_rpm,
        "chat_tpm": settings.rl_chat_tpm,
        "embed_rpm": settings.rl_embed_rpm,
        "embed_tpm": settings.rl_embed_tpm,
    }[name]
    return TokenBucket(
        key=f"rl:openai:{name.replace('_', ':')}",
        capacity=per_minute,
        refill_per_sec=per_minute / 60,
    )


def acquire_or_defer(costs: list[tuple[str, int]]) -> None:
    """Takes `cost` from each named bucket, in order, or raises.

    One definition for every stage that calls a paid API: each fix below
    was found in S3 and would otherwise have had to be found again in S4.
    """
    for bucket_name, cost in costs:
        allowed, wait_ms = get_bucket(bucket_name).acquire(cost)
        if allowed:
            continue

        if wait_ms < 0:
            # Not "wait and it clears" but "no amount of waiting clears it":
            # the request costs more than the bucket holds in total.
            # Deferring on this would compute a negative countdown, which
            # Celery turns into an ETA already in the past and so runs
            # immediately -- an endless tight loop, with nothing logged.
            raise AppException(
                ErrorCode.RATE_LIMIT_UNSATISFIABLE,
                f"{bucket_name}: costs {cost}, more than the bucket's whole capacity",
            )

        # Requeue, do NOT sleep: a sleeping worker still holds its slot and
        # reports itself busy, leaving the pool at 100% utilised while doing
        # nothing. The jitter keeps a crowd of deferred tasks from waking up
        # in lockstep and colliding on the bucket again.
        raise RateLimited(countdown=(wait_ms / 1000) * random.uniform(1.0, 1.3))
