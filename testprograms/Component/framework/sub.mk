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
	$(SRCDIR)/BuilderImpl.cc \
	$(SRCDIR)/ConnectionServicesImpl.cc \
	$(SRCDIR)/ComponentImpl.cc \
	$(SRCDIR)/ComponentIdImpl.cc \
	$(SRCDIR)/FrameworkImpl.cc \
	$(SRCDIR)/PortInfoImpl.cc \
	$(SRCDIR)/SciServicesImpl.cc \
	$(SRCDIR)/PortImpl.cc \
	$(SRCDIR)/LocalFramework.cc \
#	$(SRCDIR)/BuilderImpl.cc \
GENHDRS := $(SRCDIR)/cca_sidl.h

include $(SRCTOP)/scripts/program.mk

