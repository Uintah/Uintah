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

LIBS := $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY) $(XI_LIBRARY) $(XMU_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY)

ifeq ($(findstring -n32, $(C_FLAGS)),-n32)
#ifneq ($(USE_SOUND),no)
  PSELIBS += Packages/rtrt/Sound
  libs +=  $(SOUND_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/program.mk

# multi_rtrt
SRCS := $(SRCDIR)/multi_rtrt.cc
PROGRAM := Packages/rtrt/StandAlone/mrtrt
include $(SCIRUN_SCRIPTS)/program.mk

#nrrd2brick
SRCS := $(SRCDIR)/nrrd2brick.cc
LIBS := $(FASTM_LIBRARY) $(TEEM_LIBRARY) $(THREAD_LIBRARY) $(X11_LIBRARY) $(M_LIBRARY)
PROGRAM := Packages/rtrt/StandAlone/nrrd2brick
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
LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY)

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
LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY)

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
