#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := testprograms/Component/framework

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/CCA/Component/CIA Core/CCA/Component/PIDL Core/Thread \
	Core/Exceptions Core/globus_threads Core/Containers
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/main

SRCS    := $(SRCDIR)/cca_sidl.cc \
	$(SRCDIR)/main.cc \
	$(SRCDIR)/cca.cc \
	$(SRCDIR)/Registry.cc \
	$(SRCDIR)/ComponentIdImpl.cc \
	$(SRCDIR)/ComponentImpl.cc \
	$(SRCDIR)/PortInfoImpl.cc \
	$(SRCDIR)/PortImpl.cc \
	$(SRCDIR)/FrameworkImpl.cc \
	$(SRCDIR)/SciServicesImpl.cc \
	$(SRCDIR)/ConnectionServicesImpl.cc \
	$(SRCDIR)/RegistryServicesImpl.cc \
	$(SRCDIR)/LocalFramework.cc \
	$(SRCDIR)/Builder.cc \
	$(SRCDIR)/Sender.cc \
	$(SRCDIR)/Provider.cc \
	$(SRCDIR)/TestPortImpl.cc 

GENHDRS := $(SRCDIR)/cca_sidl.h

# Must be after the SRCS so that the REI files can be added to them.
SUBDIRS := $(SRCDIR)/REI
include $(SCIRUN_SCRIPTS)/recurse.mk

include $(SRCTOP)/scripts/program.mk

