#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Core/Datatypes

SRCS += $(SRCDIR)/Clipper.cc                    \
        $(SRCDIR)/Color.cc                      \
        $(SRCDIR)/ColumnMatrix.cc               \
        $(SRCDIR)/Datatype.cc                   \
        $(SRCDIR)/DenseColMajMatrix.cc          \
        $(SRCDIR)/DenseMatrix.cc                \
        $(SRCDIR)/Field.cc                      \
        $(SRCDIR)/Image.cc                      \
        $(SRCDIR)/Matrix.cc                     \
        $(SRCDIR)/MatrixOperations.cc           \
        $(SRCDIR)/Mesh.cc                       \
        $(SRCDIR)/PropertyManager.cc            \
        $(SRCDIR)/SearchGrid.cc                 \
        $(SRCDIR)/SparseRowMatrix.cc            \
        $(SRCDIR)/String.cc                     \
	$(SRCDIR)/FieldInterfaceAux.cc          \
        $(SRCDIR)/TypeName.cc                   \
        $(SRCDIR)/Archive.cc	                \
        $(SRCDIR)/ScalarParticles.cc 		\
        $(SRCDIR)/VectorParticles.cc 		\
        $(SRCDIR)/TensorParticles.cc 		\
        $(SRCDIR)/PSet.cc 			\
        $(SRCDIR)/VariableCache.cc 		\
#       $(SRCDIR)/cd_templates.cc \


ifeq ($(HAVE_TEEM),yes)
  SRCS += \
        $(SRCDIR)/NrrdData.cc                   
endif

PSELIBS := \
        Core/Basis        \
        Core/Containers   \
        Core/Disclosure   \
        Core/Exceptions   \
        Core/Grid   \
        Core/Geometry     \
        Core/Math         \
        Core/Persistent   \
        Core/Thread       \
        Core/Util         \
	\
	Core/ProblemSpec

LIBS := $(M_LIBRARY) $(BLAS_LIBRARY) $(F_LIBRARY)           \
        $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY)         \
        $(BLAS_LIBRARY) $(LAPACK_LIBRARY) $(THREAD_LIBRARY) \
        \
        $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(HAVE_TEEM),yes)
INCLUDES += $(TEEM_INCLUDE)
endif
INCLUDES += $(BLAS_INCLUDE)
