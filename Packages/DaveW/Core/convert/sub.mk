# Makefile fragment for this subdirectory

SRCDIR := Packages/DaveW/Core/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Packages/DaveW/Core/convert
else
PSELIBS := Core/Datatypes Packages/DaveW/ThirdParty/NumRec Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/BldDisks
SRCS := $(SRCDIR)/BldDisks.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MakeRadialCylinder
SRCS := $(SRCDIR)/MakeRadialCylinder.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/DuckReader
SRCS := $(SRCDIR)/DuckReader.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/DuckWriter
SRCS := $(SRCDIR)/DuckWriter.cc
include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/BldTensor
#SRCS := $(SRCDIR)/BldTensor.cc
#include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/CVRTItoMesh
SRCS := $(SRCDIR)/CVRTItoMesh.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/GenPlate
SRCS := $(SRCDIR)/GenPlate.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/GenSegFld
SRCS := $(SRCDIR)/GenSegFld.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/JAStoMesh
SRCS := $(SRCDIR)/JAStoMesh.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/JAStoSurf
SRCS := $(SRCDIR)/JAStoSurf.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MatrixToMat
SRCS := $(SRCDIR)/MatrixToMat.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MeshCheck
SRCS := $(SRCDIR)/MeshCheck.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MeshFix
SRCS := $(SRCDIR)/MeshFix.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MeshToJAS
SRCS := $(SRCDIR)/MeshToJAS.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MeshToSFUG
SRCS := $(SRCDIR)/MeshToSFUG.cc
include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/ModelTest
#SRCS := $(SRCDIR)/ModelTest.cc
#include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/ModelTest5
#SRCS := $(SRCDIR)/ModelTest5.cc
#include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/ModelTestFull
#SRCS := $(SRCDIR)/ModelTestFull.cc
#include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/ResampleField
SRCS := $(SRCDIR)/ResampleField.cc
include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/SFRGfile
#SRCS := $(SRCDIR)/SFRGfile.cc
#include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/SFUGtoSFUG
SRCS := $(SRCDIR)/SFUGtoSFUG.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/STreeToTris
SRCS := $(SRCDIR)/STreeToTris.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/SurfToJAS
SRCS := $(SRCDIR)/SurfToJAS.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/SurfToVDT
SRCS := $(SRCDIR)/SurfToVDT.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/VDTtoMesh
SRCS := $(SRCDIR)/VDTtoMesh.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/WatsonToMesh
SRCS := $(SRCDIR)/WatsonToMesh.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/GEtoSF
SRCS := $(SRCDIR)/GEtoSF.cc
include $(SRCTOP)/scripts/program.mk

