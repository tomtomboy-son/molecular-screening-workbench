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

class DuplicateExpectedWellError(PlateDataError):
    pass

class DuplicateVariantKeyError(PlateDataError):
    pass

class DuplicateVariantReferenceError(PlateDataError):
    pass

class MissingSequenceError(PlateDataError):
    pass

class MissingVariantReferenceError(PlateDataError):
    pass

class MissingVariantCoverageError(PlateDataError):
    pass

class MissingPlateLayoutQCError(PlateDataError):
    pass

class DuplicateHeatmapWellError(PlateDataError):
    pass

