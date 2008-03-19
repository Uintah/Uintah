#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Mixing

SRCS += \
        $(SRCDIR)/ColdflowMixingModel.cc    \
        $(SRCDIR)/InletStream.cc            \
        $(SRCDIR)/MixingModel.cc            \
        $(SRCDIR)/MOMColdflowMixingModel.cc \
        $(SRCDIR)/NewStaticMixingTable.cc   \
        $(SRCDIR)/StandardTable.cc          \
        $(SRCDIR)/Stream.cc                 

PSELIBS := \
	Packages/Uintah/Core/Exceptions       \
	Packages/Uintah/Core/IO               \
	Packages/Uintah/Core/Math             \
	Packages/Uintah/Core/ProblemSpec      \
	Packages/Uintah/Core/Util             \
	Packages/Uintah/CCA/Components/Models \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry   

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(F_LIBRARY)

#CFLAGS += -g -DARCHES_VEL_DEBUG
#CFLAGS += -g -DARCHES_DEBUG -DARCHES_GEOM_DEBUG -DARCHES_BC_DEBUG -DARCHES_COEF_DEBUG 
#CFLAGS +=
#CFLAGS += -DARCHES_SRC_DEBUG -DARCHES_PRES_DEBUG -DARCHES_VEL_DEBUG
#LIBS += $(BLAS_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


