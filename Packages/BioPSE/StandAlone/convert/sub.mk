# Makefile fragment for this subdirectory

SRCDIR := Packages/BioPSE/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/EGItoMat
SRCS := $(SRCDIR)/EGItoMat.cc
include $(SRCTOP)/scripts/program.mk
