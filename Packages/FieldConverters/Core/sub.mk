include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Packages/FieldConverters/Core

SUBDIRS := \
        $(SRCDIR)/Datatypes \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk


