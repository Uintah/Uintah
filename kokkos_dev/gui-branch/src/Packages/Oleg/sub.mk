include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Packages/Oleg

SUBDIRS := \
        $(SRCDIR)/Core \
        $(SRCDIR)/Dataflow \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk


