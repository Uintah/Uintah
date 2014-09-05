# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/rtrt/Sound

SRCS += \
	$(SRCDIR)/Sound.cc \
	$(SRCDIR)/SoundThread.cc 

PSELIBS :=  \
	Core/Thread Core/Exceptions

AUDIOFILE_LIBRARY := -L/home/sci/dav
LIBS := $(THREAD_LIBS) $(PERFEX_LIBRARY) -laudio $(AUDIOFILE_LIBRARY) -laudiofile

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
