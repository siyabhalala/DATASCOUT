"""
datascout.contracts.errors.base
────────────────────────────────
Base exception hierarchy for the DataScout system.

All custom exceptions raised anywhere in DataScout inherit from
``DataScoutBaseError``. This provides a single catch point for callers that
want to distinguish DataScout errors from unexpected third-party exceptions,
and ensures every error carries a machine-readable ``code``, a
human-readable ``message``, and an optional ``details`` dict for structured
context.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DataScoutBaseError",
    "ValidationError",
    "NotFoundError",
    "ConfigurationError",
]


class DataScoutBaseError(Exception):
    """Base class for all DataScout exceptions.

    Attributes
    ----------
    code:
        A short, uppercase, underscore-separated identifier for the error
        category (e.g. ``"VALIDATION_ERROR"``, ``"NOT_FOUND"``).  Useful for
        programmatic error handling without string matching on the message.
    message:
        A human-readable description of what went wrong.  Should be suitable
        for logging and, after scrubbing, for surfacing to end-users.
    details:
        An optional dictionary carrying structured context about the error —
        e.g. which field failed validation, which resource was not found, or
        which environment variable is missing.
    """

    def __init__(
        self,
        message: str,
        code: str = "DATASCOUT_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the exception.

        Parameters
        ----------
        message:
            Human-readable description of the error.
        code:
            Machine-readable error code.  Defaults to ``"DATASCOUT_ERROR"``.
        details:
            Optional mapping of additional structured context.
        """
        super().__init__(message)
        self.code: str = code
        self.message: str = message
        self.details: dict[str, Any] = details or {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"details={self.details!r})"
        )

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} — details: {self.details}"
        return f"[{self.code}] {self.message}"


class ValidationError(DataScoutBaseError):
    """Raised when a contract or input-validation constraint is violated.

    Use this when a ``RawDataset``, ``SearchQuery``, or any other Pydantic
    model fails validation, or when a caller passes arguments that violate a
    documented precondition.

    Example
    -------
    >>> raise ValidationError(
    ...     "row_count must be a positive integer",
    ...     details={"field": "row_count", "received": -1},
    ... )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class NotFoundError(DataScoutBaseError):
    """Raised when a requested resource cannot be located.

    Use this when a dataset lookup by canonical_id returns nothing, when a
    search history record is missing, or when any expected persistent resource
    is absent.

    Example
    -------
    >>> raise NotFoundError(
    ...     "Dataset not found",
    ...     details={"canonical_id": "kaggle:titanic"},
    ... )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="NOT_FOUND", details=details)


class ConfigurationError(DataScoutBaseError):
    """Raised when a required configuration value is absent or invalid.

    Use this for missing environment variables, invalid settings combinations,
    or misconfigured infrastructure (e.g. pointing at a non-existent
    Elasticsearch cluster when ``ELASTIC_ENABLED=true``).

    Example
    -------
    >>> raise ConfigurationError(
    ...     "KAGGLE_KEY environment variable is not set",
    ...     details={"env_var": "KAGGLE_KEY"},
    ... )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)