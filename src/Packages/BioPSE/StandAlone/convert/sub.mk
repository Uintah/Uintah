#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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

#   File   : sub.mk<2>
#   Author : Martin Cole
#   Date   : Wed Jun 20 17:45:24 2001

# Makefile fragment for this subdirectory

SRCDIR := Packages/BioPSE/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := \
	Core/Datatypes  \
	Core/Containers \
	Core/Persistent \
	Core/Exceptions \
	Core/Thread     \
	Core/Geometry   \
	Core/Math       \
	Core/Util
endif
LIBS := $(XML_LIBRARY) $(LAPACK_LIBRARY) $(M_LIBRARY)

PROGRAM := $(SRCDIR)/ContinuityToTetVolDouble
SRCS := $(SRCDIR)/ContinuityToTetVolDouble.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/EGItoMat
SRCS := $(SRCDIR)/EGItoMat.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/ElectrodeToCurve
SRCS := $(SRCDIR)/ElectrodeToCurve.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToCurve
SRCS := $(SRCDIR)/TextToCurve.cc
include $(SCIRUN_SCRIPTS)/program.mk
