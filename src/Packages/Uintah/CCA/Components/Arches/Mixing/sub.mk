#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Mixing

SRCS     += $(SRCDIR)/MixingModel.cc $(SRCDIR)/ColdflowMixingModel.cc  \
                $(SRCDIR)/DynamicTable.cc \
                $(SRCDIR)/MixRxnTableInfo.cc $(SRCDIR)/MixRxnTable.cc \
		$(SRCDIR)/PDFMixingModel.cc $(SRCDIR)/MeanMixingModel.cc \
		$(SRCDIR)/FlameletMixingModel.cc \
		$(SRCDIR)/Integrator.cc $(SRCDIR)/PDFShape.cc \
		$(SRCDIR)/BetaPDFShape.cc \
		$(SRCDIR)/ReactionModel.cc \
		$(SRCDIR)/StanjanEquilibriumReactionModel.cc \
		$(SRCDIR)/ILDMReactionModel.cc \
	        $(SRCDIR)/ChemkinInterface.cc \
                $(SRCDIR)/Common.cc \
                $(SRCDIR)/KDTree.cc  $(SRCDIR)/VectorTable.cc \
		$(SRCDIR)/Stream.cc $(SRCDIR)/InletStream.cc	

SUBDIRS := $(SRCDIR)/fortran
include $(SCIRUN_SCRIPTS)/recurse.mk
PSELIBS := \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Packages/Uintah/Core/Math          \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry   \
	Dataflow/XMLUtil

LIBS := $(PETSC_LIBRARY) $(XML_LIBRARY) $(MPI_LIBRARY) -lm $(F_LIBRARY)

#CFLAGS += -g -DARCHES_VEL_DEBUG
#CFLAGS += -g -DARCHES_DEBUG -DARCHES_GEOM_DEBUG -DARCHES_BC_DEBUG -DARCHES_COEF_DEBUG 
#CFLAGS +=
#CFLAGS += -DARCHES_SRC_DEBUG -DARCHES_PRES_DEBUG -DARCHES_VEL_DEBUG
#LIBS += -lblas

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


