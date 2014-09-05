#Makefile fragment for the Packages/Moulding directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/NetSolve
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
