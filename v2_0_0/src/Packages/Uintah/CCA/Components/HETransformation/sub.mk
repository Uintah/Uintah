# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/HETransformation

SRCS     += $(SRCDIR)/NullBurn.cc $(SRCDIR)/SimpleBurn.cc \
	$(SRCDIR)/BurnFactory.cc  $(SRCDIR)/Burn.cc \
	$(SRCDIR)/IgnitionCombustion.cc \
	$(SRCDIR)/PressureBurn.cc \


PSELIBS	:= Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
