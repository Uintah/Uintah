# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Examples

SRCS     += $(SRCDIR)/Poisson1.cc \
	$(SRCDIR)/Poisson2.cc \
	$(SRCDIR)/ExamplesLabel.cc

PSELIBS := 

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

