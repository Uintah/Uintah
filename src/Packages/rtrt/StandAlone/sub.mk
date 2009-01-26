# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

# rtrt
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/rtrt.cc

PROGRAM := Packages/rtrt/StandAlone/rtrt
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

PSELIBS := \
	Packages/rtrt/Core \
	Packages/rtrt/visinfo \
	Core/Thread \
	Core/Persistent \
	Core/Geometry \
	Core/Exceptions

endif

LIBS := $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(SOUND_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

# These need the same libray parameters as rtrt

# multi_rtrt
SRCS := $(SRCDIR)/multi_rtrt.cc
PROGRAM := Packages/rtrt/StandAlone/mrtrt
include $(SCIRUN_SCRIPTS)/program.mk

# nrrd2brick
SRCS := $(SRCDIR)/nrrd2brick.cc
LIBS += $(TEEM_LIBRARY)
PROGRAM := Packages/rtrt/StandAlone/nrrd2brick
include $(SCIRUN_SCRIPTS)/program.mk

############################################################
# These don't need the same library parameters as rtrt

# rserver
SRCS := $(SRCDIR)/rserver.cc
PROGRAM := Packages/rtrt/StandAlone/rserver
LIBS := $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY)
PSELIBS := Packages/rtrt/visinfo Core/Thread Packages/rtrt/Core
include $(SCIRUN_SCRIPTS)/program.mk

# visinfo
SRCDIR := Packages/rtrt/visinfo

SRCS := $(SRCDIR)/findvis.c

PROGRAM := Packages/rtrt/StandAlone/findvis
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/visinfo

endif
LIBS := $(GL_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

# gl
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/gl.cc

PROGRAM := Packages/rtrt/StandAlone/gl
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Thread \
	Core/Exceptions

endif
LIBS := $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(Xg_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

# glthread
SRCDIR := Packages/rtrt/StandAlone

SRCS := $(SRCDIR)/glthread.cc

PROGRAM := Packages/rtrt/StandAlone/glthread
ifeq ($(LARGESOS),yes)
  PSELIBS := Packages/rtrt
else

  PSELIBS := \
	Packages/rtrt/Core \
	Core/Thread \
	Core/Persistent \
	Core/Exceptions

endif
LIBS := $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(X_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

##################################################

SUBDIRS := \
	$(SRCDIR)/tex-utils \
	$(SRCDIR)/utils \
	$(SRCDIR)/scenes \

include $(SCIRUN_SCRIPTS)/recurse.mk

# Convenience target:
.PHONY: rtrt
rtrt: prereqs Packages/rtrt/StandAlone/rtrt scenes
.PHONY: scenes
scenes: $(SCENES)
.PHONY: librtrt
librtrt: lib/libPackages_rtrt_Core.so
