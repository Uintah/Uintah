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

# The following variables are used by the Fake* scripts... please
# do not modify...
#
#FAKE_ICE       = $(SRCDIR)/FakeICE.cc $(SRCDIR)/FakeAMRICE.cc
#FAKE_ARCHES     = $(SRCDIR)/FakeArches.cc
#FAKE_MPMARCHES  = $(SRCDIR)/FakeMPMArches.cc
#FAKE_MPM       = $(SRCDIR)/FakeMPM.cc
#FAKE_MPMICE    = $(SRCDIR)/FakeMPMICE.cc

SRCS += \
       $(FAKE_ICE) \
       $(FAKE_ARCHES) \
       $(FAKE_MPMARCHES) \
       $(FAKE_MPM) \
       $(FAKE_MPMICE)

PSELIBS := \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/CCA/Ports     \
	Packages/Uintah/CCA/Components/Solvers 

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

