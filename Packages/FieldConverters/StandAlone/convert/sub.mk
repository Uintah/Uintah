# Makefile fragment for this subdirectory

SRCDIR := Packages/FieldConverters/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core Packages/FieldConverters/Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Packages/FieldConverters/Core/Datatypes
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/OldSFRGtoNewLatticeVol
SRCS := $(SRCDIR)/OldSFRGtoNewLatticeVol.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/NewLatticeVolToOldSFRG
SRCS := $(SRCDIR)/NewLatticeVolToOldSFRG.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/OldSFUGtoNewTetVol
SRCS := $(SRCDIR)/OldSFUGtoNewTetVol.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/NewTetVolToOldSFUG
SRCS := $(SRCDIR)/NewTetVolToOldSFUG.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/OldMeshToNewTetVol
SRCS := $(SRCDIR)/OldMeshToNewTetVol.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/OldMeshToNewFieldSet
SRCS := $(SRCDIR)/OldMeshToNewFieldSet.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/NewTetVolToOldMesh
SRCS := $(SRCDIR)/NewTetVolToOldMesh.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/OldSurfaceToNewTriSurf
SRCS := $(SRCDIR)/OldSurfaceToNewTriSurf.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/NewTriSurfToOldSurface
SRCS := $(SRCDIR)/NewTriSurfToOldSurface.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/PropTest
SRCS := $(SRCDIR)/PropTest.cc
include $(SRCTOP)/scripts/program.mk
