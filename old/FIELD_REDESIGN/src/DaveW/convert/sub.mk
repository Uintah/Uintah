#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := DaveW/convert

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore
else
PSELIBS := PSECore/Datatypes DaveW/ThirdParty/NumRec SCICore/Datatypes SCICore/Containers SCICore/Persistent SCICore/Exceptions SCICore/Thread SCICore/Geometry
endif
LIBS := -lm

PROGRAM := $(SRCDIR)/BldDisks
SRCS := $(SRCDIR)/BldDisks.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM := $(SRCDIR)/MakeRadialCylinder
SRCS := $(SRCDIR)/MakeRadialCylinder.cc
include $(SRCTOP)/scripts/program.mk

#PROGRAM := $(SRCDIR)/DuckReader
#SRCS := $(SRCDIR)/DuckReader.cc
#include $(SRCTOP)/scripts/program.mk

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

#
# $Log$
# Revision 1.2.2.6  2000/11/01 23:02:31  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.4  2000/10/27 16:25:43  mcole
# fix build in convert
#
# Revision 1.2.2.3  2000/10/26 14:02:27  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.7  2000/10/30 04:39:54  dmw
# reverting to previous version
#
# Revision 1.5  2000/10/27 20:32:00  zyp
# Fixed the sub.mk file.  Accidently included one of my test programs in
# it (that I did not commit).  All better now...
#
# Revision 1.4  2000/10/24 23:58:19  zyp
# This program (DuckWriter) converts a file containing a ColumnMatrix of
# values from a forward problem solve of electrical injection into the
# head on to the readout voltage points and a file containing a
# ColumnMatrix of the injection points and creates a text file that
# hopefully is simple enough for the University of Oregon people working
# on this problem.
#
# Revision 1.3  2000/07/18 17:44:03  lfox
# added GEtoSF.cc
#
# Revision 1.2  2000/03/20 19:36:34  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:24  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
