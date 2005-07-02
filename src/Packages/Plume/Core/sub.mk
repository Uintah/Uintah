# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/Plume/Core


SRCS += 
#SUBDIRS := 

#include $(SRCTOP)/scripts/recurse.mk

PSELIBS :=  \
	Core/Thread Core/Exceptions 

LIBS := $(GLUI_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
