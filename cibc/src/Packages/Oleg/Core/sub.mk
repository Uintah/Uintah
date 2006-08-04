include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Packages/Oleg/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP)/scripts/largeso_epilogue.mk


