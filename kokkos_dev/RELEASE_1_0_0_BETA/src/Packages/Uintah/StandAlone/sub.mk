# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/StandAlone

include $(SRCTOP)/scripts/largeso_epilogue.mk

SRCS := $(SRCDIR)/sus.cc
ifneq ($(CC_DEPEND_REGEN),-MD)
# Arches doesn't work under g++ yet
ARCHES := Packages/Uintah/CCA/Components/Arches
else
SRCS := $(SRCS) $(SRCDIR)/FakeArches.cc
endif

PROGRAM := Packages/Uintah/StandAlone/sus
ifeq ($(LARGESOS),yes)
  PSELIBS := Uintah
else

  PSELIBS := \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/MPM \
	Packages/Uintah/CCA/Components/MPMICE \
	Packages/Uintah/CCA/Components/MPMArches \
	Packages/Uintah/CCA/Components/DataArchiver \
	Packages/Uintah/CCA/Components/SimulationController \
	Packages/Uintah/CCA/Components/Schedulers \
	Packages/Uintah/CCA/Components/ProblemSpecification \
	Packages/Uintah/CCA/Components/ICE \
	$(ARCHES) \
	Dataflow/Network \
	Core/Exceptions \
	Core/Thread

endif
LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(GL_LIBS)

include $(SRCTOP)/scripts/program.mk

SRCS := $(SRCDIR)/puda.cc
PROGRAM := Packages/Uintah/StandAlone/puda
ifeq ($(LARGESOS),yes)
PSELIBS := Uintah
else
PSELIBS := \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/CCA/Components/MPM \
	Dataflow/XMLUtil \
	Core/Exceptions \
	Core/Geometry \
	Core/Thread \
	Core/Util \
	Core/OS \
	Core/Containers
endif
LIBS 	:= $(XML_LIBRARY)

include $(SRCTOP)/scripts/program.mk

# A convenience target (use make sus)
sus: Packages/Uintah/StandAlone/sus
