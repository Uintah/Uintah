#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Mixing

SRCS     += $(SRCDIR)/MixingModel.cc $(SRCDIR)/ColdflowMixingModel.cc  \
                $(SRCDIR)/DynamicTable.cc \
		$(SRCDIR)/PDFMixingModel.cc $(SRCDIR)/MixRxnTableInfo.cc \
		$(SRCDIR)/Integrator.cc $(SRCDIR)/PDFShape.cc \
		$(SRCDIR)/BetaPDFShape.cc $(SRCDIR)/KDTree.cc \
		$(SRCDIR)/ReactionModel.cc \
		$(SRCDIR)/StanjanEquilibriumReactionModel.cc \
		$(SRCDIR)/ILDMReactionModel.cc \
                $(SRCDIR)/Common.cc \
		$(SRCDIR)/Stream.cc $(SRCDIR)/InletStream.cc \
		$(SRCDIR)/ChemkinInterface.cc \
		$(SRCDIR)/ILDMReactionModel.cc 

SUBDIRS := $(SRCDIR)/fortran
include $(SCIRUN_SCRIPTS)/recurse.mk
FLIB := -lftn
#CFLAGS += -g -DARCHES_VEL_DEBUG
#CFLAGS += -g -DARCHES_DEBUG -DARCHES_GEOM_DEBUG -DARCHES_BC_DEBUG -DARCHES_COEF_DEBUG 
CFLAGS +=
#CFLAGS += -DARCHES_SRC_DEBUG -DARCHES_PRES_DEBUG -DARCHES_VEL_DEBUG
#LIBS += -lblas


