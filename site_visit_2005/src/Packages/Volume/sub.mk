#Makefile fragment for the Packages/Volume directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Volume
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

#[INSERT NEW CODE FILE HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk
