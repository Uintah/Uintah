#Makefile fragment for the Packages/DaveW/Dataflow directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/DaveW/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
