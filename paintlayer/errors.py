"""Errors reported with structured field and operation context."""


class ExamError(Exception):
    status = 400
    code = 'invalid_request'

    def __init__(self, message, *, field=None, facts=None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.facts = facts or {}

    def as_dict(self):
        details = {
            'code': self.code,
            'message': self.message,
        }
        if self.field is not None:
            details['field'] = self.field
        if self.facts:
            details['facts'] = self.facts
        return details


class MissingRecord(ExamError):
    status = 404
    code = 'record_missing'


class RevisionConflict(ExamError):
    status = 409
    code = 'revision_conflict'


class CapacityExceeded(ExamError):
    status = 413
    code = 'capacity_exceeded'


class ImageFormatError(ExamError):
    status = 422
    code = 'image_format'


class InsufficientSupport(ExamError):
    status = 422
    code = 'insufficient_support'


class InvalidGeometry(ExamError):
    status = 422
    code = 'invalid_geometry'
