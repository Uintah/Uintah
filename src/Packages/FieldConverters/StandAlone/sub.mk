#Makefile fragment for the Packages/FieldConverters directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/FieldConverters/StandAlone
SUBDIRS := \
	$(SRCDIR)/convert \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
