add_library( Core_Exceptions
        ArrayIndexOutOfBounds.cc
        AssertionFailed.cc
        ConvergenceFailure.cc
        DimensionMismatch.cc
        ErrnoException.cc
        Exception.cc
        FileNotFound.cc
        InternalError.cc
        InvalidCompressionMode.cc
        InvalidGrid.cc
        InvalidState.cc
        InvalidValue.cc
        NotYetImplemented.cc
        ParameterNotFound.cc
        ProblemSetupException.cc
        TypeMismatchException.cc
        UintahPetscError.cc
        VariableNotFoundInGrid.cc
    )

target_link_libraries( Core_Exceptions
        ${CMAKE_DL_LIBS}
    )

target_include_directories( Core_Exceptions
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

install( FILES
    ArrayIndexOutOfBounds.h
    AssertionFailed.h
    ConvergenceFailure.h
    DimensionMismatch.h
    ErrnoException.h
    Exception.h
    FileNotFound.h
    IllegalValue.h
    InternalError.h
    InvalidCompressionMode.h
    InvalidGrid.h
    InvalidState.h
    InvalidValue.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Core/Exceptions
)

install( TARGETS Core_Exceptions
    EXPORT uintah_lib_targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
