# Makefile fragment for this subdirectory

SRCDIR := testprograms/Core/CCA/Component/pingpong

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Component/CIA Component/PIDL Core/Thread \
	Core/Exceptions Core/globus_threads
endif
LIBS := $(GLOBUS_LIBS) -lglobus_nexus -lglobus_dc -lglobus_common -lglobus_io

PROGRAM := $(SRCDIR)/pingpong
SRCS := $(SRCDIR)/pingpong.cc $(SRCDIR)/PingPong_sidl.cc \
	$(SRCDIR)/PingPong_impl.cc
GENHDRS := $(SRCDIR)/PingPong_sidl.h

include $(SRCTOP)/scripts/program.mk

