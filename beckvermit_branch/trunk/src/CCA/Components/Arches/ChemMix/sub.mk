# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/ChemMix

SRCS += \
        $(SRCDIR)/ClassicTableInterface.cc \
        $(SRCDIR)/ColdFlow.cc              \
        $(SRCDIR)/ConstantProps.cc         \
        $(SRCDIR)/MixingRxnModel.cc        \
        $(SRCDIR)/PCTProps.cc

ifeq ($(HAVE_TABPROPS),yes)
   SRCS += $(SRCDIR)/TabPropsInterface.cc
endif

PSELIBS := $(PSELIBS) Core/IO
