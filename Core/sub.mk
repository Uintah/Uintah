#Makefile fragment for the Packages/Uintah/Core directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Uintah/Core
SUBDIRS := \
	$(SRCDIR)/Parallel \
	$(SRCDIR)/Math \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/Grid \
	$(SRCDIR)/ProblemSpec \
	$(SRCDIR)/Datatypes

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
