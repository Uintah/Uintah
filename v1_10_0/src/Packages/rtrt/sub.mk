#Makefile fragment for the Packages/rtrt directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/rtrt

SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/Dataflow     \
	$(SRCDIR)/visinfo

ifeq ($(HAVE_AUDIO),yes)
SUBDIRS += $(SRCDIR)/Sound
endif

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := Core
LIBS := $(OOGL_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) 

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

SUBDIRS := \
	$(SRCDIR)/StandAlone
include $(SCIRUN_SCRIPTS)/recurse.mk

#CFLAGS := $(CFLAGS) -OPT:IEEE_arithmetic=3 
#CFLAGS := $(CFLAGS) -OPT:Olimit=16383
#CFLAGS := $(CFLAGS) -OPT:Olimit_opt=on
