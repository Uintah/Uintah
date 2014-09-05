#Makefile fragment for the Packages/Netsolve/Core directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/NetSolve/Core

SUBDIRS := \
	$(SRCDIR)/Datatypes\

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk
