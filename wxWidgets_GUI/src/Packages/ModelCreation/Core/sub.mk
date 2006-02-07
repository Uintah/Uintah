include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Packages/ModelCreation/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \
        $(SRCDIR)/Algorithms \
        $(SRCDIR)/Fields \
        $(SRCDIR)/Util \

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk


