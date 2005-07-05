# Makefile fragment for this subdirectory
#

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/CCA/Components/Dummy

SRCS += $(SRCDIR)/FakeArches.cc    \
	$(SRCDIR)/FakeICE.cc       \
	$(SRCDIR)/FakeMPM.cc       \
	$(SRCDIR)/FakeMPMArches.cc \
	$(SRCDIR)/FakeMPMICE.cc 

PSELIBS := 

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

