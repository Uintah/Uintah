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
        ParameterNotFound.cc
        ProblemSetupException.cc
        TypeMismatchException.cc
        UintahPetscError.cc
        VariableNotFoundInGrid.cc
    )

# jcs needs "DL_LIBRARY" ?
#target_link_libraries( Core_Exceptions
#        Core_Geometry
#    )

target_include_directories( Core_Exceptions
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )