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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Events

SRCS     += \
	$(SRCDIR)/BaseEvent.cc		        \
	$(SRCDIR)/DataManager.cc	        \
	$(SRCDIR)/EventManager.cc	        \
	$(SRCDIR)/OpenGLViewer.cc	        \
	$(SRCDIR)/SceneGraphEvent.cc            \
	$(SRCDIR)/Trail.cc	        \
	$(SRCDIR)/SelectionTargetEvent.cc	

ifeq ($(IS_WIN),yes)
  SRCS +=$(SRCDIR)/Win32EventSpawner.cc
else
  SRCS +=$(SRCDIR)/X11EventSpawner.cc	
  SRCS +=$(SRCDIR)/OSXEventSpawner.cc	
endif


SUBDIRS := $(SRCDIR)/Tools
include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := Core/Algorithms/Fields \
           Core/Algorithms/Visualization \
	   Core/Containers  \
           Core/Datatypes \
           Core/Exceptions \
	   Core/Geom  \
	   Core/Geometry  \
	   Core/Math  \
	   Core/Persistent  \
	   Core/Thread \
	   Core/Util

ifeq ($(IS_WIN),yes)
  PSELIBS += Core_OS
endif

LIBS := $(THREAD_LIBRARY) $(MPEG_LIBRARY) $(GL_LIBRARY) $(GL_LIBRARY) $(PNG_LIBRARY) 

ifeq ($(IS_OSX),yes)
  LIBS += -framework AGL -framework Carbon
endif


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

