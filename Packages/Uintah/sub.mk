#Makefile fragment for the Packages/Uintah directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Uintah
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
