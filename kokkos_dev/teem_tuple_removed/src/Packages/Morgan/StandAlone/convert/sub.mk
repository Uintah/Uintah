# Makefile fragment for this subdirectory

SRCDIR := Packages/Morgan/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Packages/Morgan/StandAlone/convert
else
PSELIBS := Core/Datatypes Core/Math Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := $(M_LIBRARY)

PROGRAM := $(SRCDIR)/BuildNeurons
SRCS := $(SRCDIR)/BuildNeurons.cc
include $(SRCTOP)/scripts/program.mk
