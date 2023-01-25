add_library( Core_Math
        CubeRoot.cc
        CubicPolyRoots.cc
        Expon.cc
        FastMatrix.cc
        fft.c
        Gaussian.cc
        Int130.cc
        LinAlg.c
        Mat.c
        Matrix3.cc
        MiscMath.cc
        MusilRNG.cc
        Primes.cc
        Rand48.cc
        sci_lapack.cc
        Short27.cc
        Sparse.cc
        ssmult.c
        SymmMatrix3.cc
        TangentModulusTensor.cc
        TrigTable.cc
        Weibull.cc
    )

target_link_libraries( Core_Math
        Core_Containers
        Core_Disclosure
        Core_Exceptions
        Core_Malloc
        Core_Util
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        m
    )

target_include_directories( Core_Math
        PUBLIC
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_BINARY_DIR}
        PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
            ${LIBXML2_INCLUDE_DIRS}
    )

if( ENABLE_FORTRAN )
  include( FortranCInterface )
  FortranCInterface_VERIFY( CXX )
  # This will generate a header with appropriate name-mangling macros for C-Fortran compatibility
  FortranCInterface_HEADER( LapackFortranNameMangle.h
      SYMBOLS
        DGEEV
        DGESV
        DGESVD
        DGETRF
        DGETRI
        DSYEV
      )
endif()