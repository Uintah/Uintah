#Makefile fragment for the Packages/Yarden/Core directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Yarden/Core
SUBDIRS := \
	$(SRCDIR)/Algorithms \
	$(SRCDIR)/Datatypes \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
