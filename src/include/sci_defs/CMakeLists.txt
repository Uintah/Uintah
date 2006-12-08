#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# DV figure out mandatory sci_defs and finish

# configure some headers files to pass some of the CMake settings to the
# source code


CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/blas_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/blas_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/bits_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/bits_defs.h"
)

# DV Fix chromium (DEF_CHROMIUM not set)
CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/chromium_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/chromium_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/compile_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/compile_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/cmake_config_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/config_defs.h"
)

# DV Fix collab_vis (DEF_COLLAB_VIS not set)
CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/collab_vis_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/collab_vis_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/cmake_config_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/config_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/error_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/error_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/cmake_environment_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/environment_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/exe_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/exe_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/hashmap_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/hashmap_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/image_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/image_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/lapack_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/lapack_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/malloc_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/malloc_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/ogl_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/ogl_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/petsc_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/petsc_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/ptolemy_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/ptolemy_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/template_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/template_defs.h"
)

CONFIGURE_FILE (
  "${PROJECT_SOURCE_DIR}/include/sci_defs/thread_testdefs.h.in"
  "${PROJECT_BINARY_DIR}/include/sci_defs/thread_defs.h"
)

CONFIGURE_FILE (
      "${PROJECT_SOURCE_DIR}/include/sci_defs/x11_testdefs.h.in"
      "${PROJECT_BINARY_DIR}/include/sci_defs/x11_defs.h"
)


#---------------------------------------------------------------------
# Package configure files
CONFIGURE_FILE (
      "${PROJECT_SOURCE_DIR}/include/sci_defs/teem_testdefs.h.in"
      "${PROJECT_BINARY_DIR}/include/sci_defs/teem_defs.h"
)

CONFIGURE_FILE (
      "${PROJECT_SOURCE_DIR}/include/sci_defs/insight_testdefs.h.in"
      "${PROJECT_BINARY_DIR}/include/sci_defs/insight_defs.h"
)

