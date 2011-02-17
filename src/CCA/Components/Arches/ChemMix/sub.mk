# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/ChemMix

SRCS += \
        $(SRCDIR)/MixingRxnModel.cc \
        $(SRCDIR)/ClassicTableInterface.cc \
				$(SRCDIR)/ColdFlow.cc \
        $(SRCDIR)/TabPropsInterface.cc 

PSELIBS := $(PSELIBS) Core/IO
