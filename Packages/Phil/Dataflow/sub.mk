#Makefile fragment for the Packages/Phil/Dataflow directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Phil/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
