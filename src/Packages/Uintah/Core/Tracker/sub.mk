# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Tracker

SRCS += \
	$(SRCDIR)/Tracker.cc       \
	$(SRCDIR)/TrackerClient.cc \
	$(SRCDIR)/TrackerServer.cc 

PSELIBS := \
	Core/Exceptions               \
	Core/Thread                   \
	Core/Util                     

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

