#
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


# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Ports

SRCS     += $(SRCDIR)/BundlePort.cc		  \
	    $(SRCDIR)/ColorMapPort.cc             \
	    $(SRCDIR)/Colormap2Port.cc            \
            $(SRCDIR)/FieldPort.cc                \
            $(SRCDIR)/GeometryPort.cc             \
	    $(SRCDIR)/ImagePort.cc		  \
            $(SRCDIR)/MatrixPort.cc               \
            $(SRCDIR)/NrrdPort.cc	          \
            $(SRCDIR)/PathPort.cc                 \
            $(SRCDIR)/TexturePort.cc              \
            $(SRCDIR)/TimePort.cc                 \
            $(SRCDIR)/templates.cc		  \


PSELIBS := Dataflow/Network Dataflow/Comm Core/Containers \
	Core/Thread Core/Geom Core/Geometry Core/Exceptions Core/Util \
	Core/Persistent Core/Datatypes Core/Volume Core/Bundle

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif
