#Makefile fragment for the Packages/Nrrd/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Nrrd/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
