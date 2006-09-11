# DV figure out mandatory sci_defs and finish

# configure some headers files to pass some of the CMake settings to the
# source code


CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/blas_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/blas_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/bits_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/bits_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/compile_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/compile_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/cmake_config_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/config_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/error_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/error_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/environment_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/environment_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/exe_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/exe_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/hashmap_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/hashmap_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/image_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/image_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/lapack_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/lapack_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/malloc_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/malloc_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/ogl_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/ogl_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/template_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/template_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/src/include/sci_defs/thread_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/thread_defs.h"
)

#---------------------------------------------------------------------
# Package configure files
CONFIGURE_FILE (
      "${PROJECT_SOURCE_DIR}/src/include/sci_defs/teem_testdefs.h.in"
      "${PROJECT_BINARY_DIR}/include/sci_defs/teem_defs.h"
)

CONFIGURE_FILE (
      "${PROJECT_SOURCE_DIR}/src/include/sci_defs/insight_testdefs.h.in"
      "${PROJECT_BINARY_DIR}/include/sci_defs/insight_defs.h"
)

