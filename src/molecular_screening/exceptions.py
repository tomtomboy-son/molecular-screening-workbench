class PlateDataError(Exception):
    pass

class MissingRequiredColumnsError(PlateDataError):
    pass

class InvalidWellError(PlateDataError):
    pass

class InvalidSignalError(PlateDataError):
    pass

class DuplicateWellError(PlateDataError):
    pass

class MissingControlError(PlateDataError):
    pass
