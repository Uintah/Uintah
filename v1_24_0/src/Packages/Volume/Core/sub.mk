#Makefile fragment for the Packages/Volume/Core directory

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Volume/Core

SUBDIRS :=  \
	$(SRCDIR)/Algorithms  \
	$(SRCDIR)/Datatypes  \
	$(SRCDIR)/Geom      \
	$(SRCDIR)/Util  \
#[INSERT NEW CODE FILE HERE]

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk
