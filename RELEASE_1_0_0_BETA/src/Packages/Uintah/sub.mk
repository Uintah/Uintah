#Makefile fragment for the Packages/Uintah directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Uintah
SUBDIRS := \
	$(SRCDIR)/Core         \
	$(SRCDIR)/Dataflow     \
	$(SRCDIR)/CCA          \
	$(SRCDIR)/StandAlone   \
	$(SRCDIR)/testprograms

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
