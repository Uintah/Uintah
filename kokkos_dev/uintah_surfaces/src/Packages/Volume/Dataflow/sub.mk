#Makefile fragment for the Packages/Volume/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Volume/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \
	$(SRCDIR)/Ports

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk

ifeq ($(LARGESOS),yes)
VOLUME_MODULES := $(VOLUME_MODULES) $(LIBNAME)
endif

volumemodules: prereqs $(VOLUME_MODULES)
