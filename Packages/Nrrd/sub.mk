#Makefile fragment for the Packages/Nrrd directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Nrrd
SUBDIRS := \
	$(SRCDIR)/Core \
	$(SRCDIR)/Dataflow \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(NRRD_LIBRARY) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
