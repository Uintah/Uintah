#Makefile fragment for the Packages/Nrrd/Core directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Nrrd/Core
SUBDIRS := \
	$(SRCDIR)/Datatypes \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(NRRD_LIBRARY) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
