#Makefile fragment for the Packages/DaveW directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/DaveW
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \
	$(SRCDIR)/StandAlone \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
