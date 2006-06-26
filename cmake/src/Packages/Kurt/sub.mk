#Makefile fragment for the Packages/Kurt directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Kurt
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \
#	$(SRCDIR)/StandAlone \
#[INSERT NEW CODE FILE HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk
