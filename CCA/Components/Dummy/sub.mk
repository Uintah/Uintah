# Makefile fragment for this subdirectory
#

#
# The Dummy components are used when the user does not wish to compile
# the real component.  They allow for the linker to satisfy the need
# for all of the components.  However, if a dummy component is used
# (accidentally) in a 'simulation', then an error message is printed
# out so that the user can know that he or she is using a dummy
# component and not the real one.
#

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/CCA/Components/Dummy

SRCS += $(SRCDIR)/FakeArches.cc    \

#	$(SRCDIR)/FakeICE.cc       \
#	$(SRCDIR)/FakeMPM.cc       \
#	$(SRCDIR)/FakeMPMArches.cc \
#	$(SRCDIR)/FakeMPMICE.cc 

PSELIBS := 

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

