#Makefile fragment for the Packages/Butson/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Butson/Dataflow
SUBDIRS := \
	$(SRCDIR)/Modules \
	$(SRCDIR)/GUI \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
