# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Algorithms/Visualization

SRCS     += $(SRCDIR)/Screen.cc 

PSELIBS := Core/Datatypes Dataflow/Network \
	   Core/Persistent Core/Containers Core/Util \
	   Core/Exceptions Core/Thread Core/TclInterface \
	   Core/Geom Core/Datatypes Core/Geometry \
	   Core/TkExtensions

LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

