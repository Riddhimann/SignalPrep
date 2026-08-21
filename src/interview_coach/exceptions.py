class InterviewCoachError(Exception):
    """Base controlled application error."""


class ConfigurationError(InterviewCoachError):
    pass


class DocumentError(InterviewCoachError):
    pass


class AudioError(InterviewCoachError):
    pass


class ModelUnavailableError(InterviewCoachError):
    pass


class StructuredOutputError(InterviewCoachError):
    pass


class SessionNotFoundError(InterviewCoachError):
    pass


class InvalidSessionStateError(InterviewCoachError):
    pass
