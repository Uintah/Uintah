#Makefile fragment for the Packages/Uintah/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Uintah/CCA
SUBDIRS := \
	$(SRCDIR)/Components \
	$(SRCDIR)/Ports \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS :=

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
