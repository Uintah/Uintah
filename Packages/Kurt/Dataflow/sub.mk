#Makefile fragment for the Packages/Kurt/Dataflow directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Kurt/Dataflow
SUBDIRS := \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Modules \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk

ifeq ($(LARGESOS),yes)
KURT_MODULES := $(KURT_MODULES) $(LIBNAME)
endif

kurtmodules: prereqs $(KURT_MODULES)

