include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/Insight/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \
        $(SRCDIR)/CodeGenerator \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk


