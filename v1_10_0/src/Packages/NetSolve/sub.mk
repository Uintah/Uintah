#Makefile fragment for the Packages/Moulding directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/NetSolve
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk
