# Makefile fragment for this subdirectory

SRCDIR := Packages/CardioWave/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Packages/CardioWave/StandAlone/convert
else
PSELIBS := Core/Datatypes Core/Math Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/CardioWaveToColumnMat
SRCS := $(SRCDIR)/CardioWaveToColumnMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CardioWaveToDenseMat
SRCS := $(SRCDIR)/CardioWaveToDenseMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/ColumnMatToCardioWave
SRCS := $(SRCDIR)/ColumnMatToCardioWave.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/SubsetPts
SRCS := $(SRCDIR)/SubsetPts.cc
include $(SRCTOP)/scripts/program.mk
