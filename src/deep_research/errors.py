"""Public errors contain stable categories, never raw upstream payloads."""


class ResearchError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ConfigurationError(ResearchError):
    pass


class ModelError(ResearchError):
    def __init__(
        self, code: str, message: str, *, status_code: int | None = None, response_usage=None
    ) -> None:
        super().__init__(code, message, status_code=status_code)
        # Error envelopes may still contain known usage (for example truncation).
        # Keep only the validated numeric usage, never raw upstream payloads.
        from deep_research.schemas import TokenUsage

        if response_usage is not None and not isinstance(response_usage, TokenUsage):
            raise TypeError("response_usage must be TokenUsage or None")
        self.response_usage = (
            TokenUsage.model_validate(response_usage.model_dump())
            if response_usage is not None
            else None
        )


class StructuredOutputError(ResearchError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        validation: dict | None = None,
    ) -> None:
        super().__init__(code, message, status_code=status_code)
        from deep_research.structured_diagnostics import validate_diagnostic

        self.validation = validate_diagnostic(validation) if validation is not None else None
