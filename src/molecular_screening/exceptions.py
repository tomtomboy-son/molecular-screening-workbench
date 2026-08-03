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

class MissingPlateQCError(PlateDataError):
    pass

class DuplicatePlateQCError(PlateDataError):
    pass

class InvalidNormalizationDenominatorError(PlateDataError):
    pass

class MissingVariantIDError(PlateDataError):
    pass
