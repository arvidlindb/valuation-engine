"""Exceptions raised by the data layer."""

from __future__ import annotations


class DataError(Exception):
    """Base class for all data-layer errors."""


class DataUnavailableError(DataError):
    """The requested data cannot be provided.

    Raised when offline mode is active and neither the fixtures nor the local
    cache contain the data, when the network fails and there is no local
    fallback, or when Yahoo has no usable data for a ticker (no price, no
    financial statements, ...). The message states which of these applies.
    """


class NetworkDisabledError(DataError):
    """A network call was attempted although offline mode is active.

    This is a safety net: the loaders never call the network layer in offline
    mode, so seeing this error indicates a programming mistake.
    """


class FetchError(DataError):
    """A network request to Yahoo Finance failed after all retries."""


class RateLimitError(FetchError):
    """Yahoo Finance rejected the request because of rate limiting."""
