class ParserError(Exception):
    def __init__(self, message: str, filename: str | None = None) -> None:
        self.message = message
        self.filename = filename
        super().__init__(message)


class FileBlockedError(Exception):
    def __init__(self, message: str, reason: str | None = None) -> None:
        self.message = message
        self.reason = reason
        super().__init__(message)


class FileSecurityTimeoutError(Exception):
    pass


class FileSecurityUnavailableError(Exception):
    pass


class FileSecurityProtocolError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)
