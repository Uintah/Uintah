# http://software.sci.utah.edu/doc/Developer/Guide/create_module.html
#
# Makefile fragment for this subdirectory
# $Id: sub.mk 40521 2008-03-19 21:06:39Z dav $
#
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/CCA/Components/Arches/TabProps

SRCS := \
        $(SRCDIR)/BSpline.cc \
        $(SRCDIR)/LU.cc \
        $(SRCDIR)/StateTable.cc

PSELIBS := \
    Core/Geometry

# variables HDF5_LIBRARY, BLAS_LIBRARY, etc are set in configVars.mk
# everything compiles OK with either LIBS var, I don't think the blas/lapack libraries make a difference
# (maybe check with BLAS and LAPACK libraries soon - use w/ DQMOM)

#LIBS := $(HDF5_LIBRARY) $(BLAS_LIBRARY) $(LAPACK_LIBRARY)
LIBS := $(HDF5_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

