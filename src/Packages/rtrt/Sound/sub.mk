# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/rtrt/Sound

SRCS += \
	$(SRCDIR)/Sound.cc \
	$(SRCDIR)/SoundThread.cc 

PSELIBS :=  \
	Core/Thread Core/Exceptions

LIBS := $(THREAD_LIBS) $(PERFEX_LIBRARY) -laudio -laudiofile

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
