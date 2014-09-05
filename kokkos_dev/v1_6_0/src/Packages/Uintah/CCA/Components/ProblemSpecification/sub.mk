# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/ProblemSpecification

SRCS	+= $(SRCDIR)/ProblemSpecReader.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid \
	Dataflow/XMLUtil
LIBS 	:= $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#PROGRAM	:= $(SRCDIR)/testing
#SRCS	:= $(SRCDIR)/testing.cc
#include $(SCIRUN_SCRIPTS)/program.mk
#PROGRAM	:= $(SRCDIR)/test2
#SRCS	:= $(SRCDIR)/test2.cc 
#ifeq ($(LARGESOS),yes)
#  PSELIBS := Packages/Uintah
#else
#  PSELIBS := \
#	Packages/Uintah/CCA/Ports \
#	Packages/Uintah/Core/Grid \
#	Packages/Uintah/Core/ProblemSpec \
#	Packages/Uintah/CCA/Components/ProblemSpecification \
#	Dataflow/XMLUtil
#endif
#LIBS 	:= $(XML_LIBRARY)
#include $(SCIRUN_SCRIPTS)/program.mk


