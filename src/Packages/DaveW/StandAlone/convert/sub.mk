# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Packages/DaveW/StandAlone/convert
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/GenTestField
SRCS := $(SRCDIR)/GenTestField.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/JAStoMesh
SRCS := $(SRCDIR)/JAStoMesh.cc
include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/MeshToJAS
#SRCS := $(SRCDIR)/MeshToJAS.cc
#include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/RawToSurface
#SRCS := $(SRCDIR)/RawToSurface.cc
#include $(SRCTOP)/scripts/program.mk

