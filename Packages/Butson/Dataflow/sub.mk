#Makefile fragment for the Packages/Butson/Dataflow directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Butson/Dataflow
SUBDIRS := \
	$(SRCDIR)/Modules \
#	$(SRCDIR)/GUI \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
