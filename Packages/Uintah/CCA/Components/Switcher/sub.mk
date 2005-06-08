# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR  := Packages/Uintah/CCA/Components/Switcher

SRCS    := $(SRCDIR)/Switcher.cc

PSELIBS := \
        Packages/Uintah/CCA/Ports        \
        Packages/Uintah/Core/Exceptions  \
        Packages/Uintah/Core/Parallel    \
        Packages/Uintah/Core/ProblemSpec \
        Packages/Uintah/Core/Util

LIBS    := $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
