# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/rtrt/visinfo


SRCS += $(SRCDIR)/visinfo.c

PSELIBS :=

LIBS := $(GL_LIBS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


