#Makefile fragment for the Packages/Yarden/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Yarden/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
