# Makefile fragment for this subdirectory


SRCDIR   := Packages/Uintah/Core/ProblemSpec

SRCS     += \
	$(SRCDIR)/ProblemSpec.cc 

PSELIBS := \
	Core/Exceptions \
	Core/Thread

LIBS := $(XML_LIBRARY) 


