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
	Core/Exceptions

endif
LIBS := $(GL_LIBS) -lfastm -lm -lelf -lfetchop -lperfex

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
LIBS := $(GL_LIBS)

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
LIBS := $(GL_LIBS) -lfastm -lm -lelf -lfetchop -lperfex

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
	Core/Exceptions

endif
LIBS := $(GL_LIBS) -lfastm -lm -lelf -lfetchop -lperfex

include $(SCIRUN_SCRIPTS)/program.mk


# A convenience target (use make rtrt)
rtrt: Packages/rtrt/StandAlone/rtrt

SUBDIRS := \
	Packages/rtrt/StandAlone/scenes
include $(SCIRUN_SCRIPTS)/recurse.mk
