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

LIBS := $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(SOUND_LIBRARY) -L/home/collaborator/dghosh/ -lpng -lz

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
LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(Xg_LIBRARY)

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
LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(X_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk


SUBDIRS := \
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
