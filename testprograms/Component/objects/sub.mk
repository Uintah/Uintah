# Makefile fragment for this subdirectory

SRCDIR := testprograms/Component/objects

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/CCA/Component/CIA Core/CCA/Component/PIDL Core/Thread \
	Core/Exceptions Core/globus_threads
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/objects
SRCS := $(SRCDIR)/objects.cc $(SRCDIR)/objects_sidl.cc
GENHDRS := $(SRCDIR)/objects_sidl.h

include $(SRCTOP)/scripts/program.mk

