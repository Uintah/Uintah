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

SRCDIR   := Core/2d

SRCS     += \
	$(SRCDIR)/Axes.cc \
	$(SRCDIR)/AxesObj.cc \
	$(SRCDIR)/BBox2d.cc \
	$(SRCDIR)/BoxObj.cc \
	$(SRCDIR)/Diagram.cc \
	$(SRCDIR)/DrawGui.cc \
	$(SRCDIR)/DrawObj.cc \
	$(SRCDIR)/Graph.cc \
	$(SRCDIR)/HairObj.cc \
	$(SRCDIR)/Hairline.cc \
	$(SRCDIR)/HistObj.cc \
	$(SRCDIR)/OpenGL.cc \
	$(SRCDIR)/OpenGLWindow.cc \
	$(SRCDIR)/Point2d.cc \
	$(SRCDIR)/Polyline.cc \
	$(SRCDIR)/ParametricPolyline.cc \
	$(SRCDIR)/LockedPolyline.cc \
	$(SRCDIR)/ScrolledOpenGLWindow.cc \
	$(SRCDIR)/Vector2d.cc \
	$(SRCDIR)/Widget.cc \
	$(SRCDIR)/Zoom.cc \
	$(SRCDIR)/glprintf.cc \
	$(SRCDIR)/asciitable.cc \
	$(SRCDIR)/texture.cc \

PSELIBS := Core/Persistent Core/Exceptions \
	Core/Math Core/Containers Core/Thread \
	Core/GuiInterface Core/Datatypes Core/Geom \
	Core/Util Core/TkExtensions
LIBS := $(TCL_LIBRARY) $(GL_LIBRARY) $(TK_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

