#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := testprograms/Component/framework

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/Thread \
	Core/Exceptions Core/CCA/Comm Core/Containers
endif

ifeq ($(HAVE_GLOBUS),yes)
PSELIBS+=Core/globus_threads
endif

LIBS := 

PROGRAM := $(SRCDIR)/main

SRCS    := \
	$(SRCDIR)/cca_sidl.cc \
	$(SRCDIR)/main.cc \
	$(SRCDIR)/cca.cc \
	$(SRCDIR)/Registry.cc \
	$(SRCDIR)/ComponentIdImpl.cc \
	$(SRCDIR)/ComponentImpl.cc \
	$(SRCDIR)/PortInfoImpl.cc \
	$(SRCDIR)/PortImpl.cc \
	$(SRCDIR)/FrameworkImpl.cc \
	$(SRCDIR)/SciServicesImpl.cc \
	$(SRCDIR)/BuilderServicesImpl.cc \
	$(SRCDIR)/RegistryServicesImpl.cc \
	$(SRCDIR)/LocalFramework.cc \
	$(SRCDIR)/TestPortImpl.cc 

GENHDRS := $(SRCDIR)/cca_sidl.h

# Must be after the SRCS so that the REI files can be added to them.
SUBDIRS := $(SRCDIR)/Builders $(SRCDIR)/REI $(SRCDIR)/TestComponents
include $(SCIRUN_SCRIPTS)/recurse.mk

include $(SRCTOP)/scripts/program.mk

