include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/DDDAS/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk


