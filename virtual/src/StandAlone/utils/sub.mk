#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Makefile fragment for this subdirectory

SRCDIR := StandAlone/utils

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := \
	Core/Datatypes Core/Util Core/Containers Core/Persistent \
	Core/Exceptions Core/Thread Core/Geometry Core/Math Core/Geom \
	Core/Init Core/Basis

endif
LIBS := $(LAPACK_LIBRARY) $(XML_LIBRARY) $(M_LIBRARY)

PROGRAM := $(SRCDIR)/PCA-example
SRCS := $(SRCDIR)/PCA-example.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/MaskLatVolWithHexVol
SRCS := $(SRCDIR)/MaskLatVolWithHexVol.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/SwapFaces
SRCS := $(SRCDIR)/SwapFaces.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/AddTri
SRCS := $(SRCDIR)/AddTri.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RemoveFaces
SRCS := $(SRCDIR)/RemoveFaces.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RemoveConnectedFaces
SRCS := $(SRCDIR)/RemoveConnectedFaces.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/RemoveOrphanNodes
SRCS := $(SRCDIR)/RemoveOrphanNodes.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/FieldBin1Test
SRCS := $(SRCDIR)/FieldBin1Test.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/FieldTextToBin
SRCS := $(SRCDIR)/FieldTextToBin.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/GenerateMPMData
SRCS := $(SRCDIR)/GenerateMPMData.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/OrientFaces
SRCS := $(SRCDIR)/OrientFaces.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TestBasis
SRCS := $(SRCDIR)/TestBasis.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/UnitElementMesh
SRCS := $(SRCDIR)/UnitElementMesh.cc
include $(SCIRUN_SCRIPTS)/program.mk


