# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR	:= Packages/Uintah/CCA/Components/Switcher

SRCS	:= $(SRCDIR)/Switcher.cc

PSELIBS := \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Util \
	Packages/Uintah/CCA/Ports

LIBS 	:= $(XML_LIBRARY) $(MPI_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
