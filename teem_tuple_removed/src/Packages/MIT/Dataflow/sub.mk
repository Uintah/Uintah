include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/MIT/Dataflow

SUBDIRS := \
        $(SRCDIR)/GUI \
        $(SRCDIR)/Modules \
        $(SRCDIR)/Ports \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk


