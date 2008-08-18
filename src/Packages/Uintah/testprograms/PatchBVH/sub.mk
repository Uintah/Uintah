
# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/testprograms/PatchBVH

PROGRAM := $(SRCDIR)/PatchBVHTest
SRCS := $(SRCDIR)/PatchBVHTest.cc

PSELIBS := \
        Core/Exceptions                          \
        Core/Geometry                            \
        Packages/Uintah/Core/Grid                \
        Packages/Uintah/Core/Util                

LIBS := $(BLAS_LIBRARY) $(LAPACK_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

