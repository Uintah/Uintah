#Makefile fragment for the Packages/McQ directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/McQ
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
