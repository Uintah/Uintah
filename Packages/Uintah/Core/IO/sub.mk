# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Packages/Uintah/Core/IO

SRCS += \
	$(SRCDIR)/UintahZlibUtil.cc

PSELIBS :=
LIBS    := $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

