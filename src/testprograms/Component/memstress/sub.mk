# Makefile fragment for this subdirectory

SRCDIR := testprograms/Core/CCA/Component/memstress

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Component/CIA Component/PIDL Core/Thread \
	Core/Exceptions Core/globus_threads
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/memstress
SRCS := $(SRCDIR)/memstress.cc $(SRCDIR)/memstress_sidl.cc
GENHDRS := $(SRCDIR)/memstress_sidl.h

include $(SRCTOP)/scripts/program.mk

