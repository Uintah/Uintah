add_library( Core_Datatypes
        ColumnMatrix.cc
        Datatype.cc
        DenseColMajMatrix.cc
        DenseMatrix.cc
        Matrix.cc
        MatrixOperations.cc
        SparseRowMatrix.cc
        TypeName.cc
    )

target_link_libraries( Core_Datatypes
    PRIVATE
        Core_Containers
        Core_Malloc
        Core_Math
        Core_Util
        ${BLAS_LIBRARIES}
    )

target_include_directories( Core_Datatypes
    PUBLIC
        ${CMAKE_SOURCE_DIR}
    )
