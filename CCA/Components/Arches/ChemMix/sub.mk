# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/ChemMix

SRCS += \
        $(SRCDIR)/MixingRxnModel.cc \
        $(SRCDIR)/ClassicTableInterface.cc \
				$(SRCDIR)/ColdFlow.cc 

ifeq ($(HAVE_TABPROPS_FIX),yes)
	SRCS += $(SRCDIR)/TabPropsInterface.cc
endif

PSELIBS := $(PSELIBS) Core/IO
