#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/convert

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore
else
PSELIBS := SCICore/Datatypes SCICore/Containers SCICore/Persistent SCICore/Exceptions SCICore/Thread SCICore/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/BldDisks
SRCS := $(SRCDIR)/BldDisks.cc
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

#
# $Log$
# Revision 1.2  2000/03/20 19:36:34  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:24  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
