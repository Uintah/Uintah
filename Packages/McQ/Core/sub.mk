#Makefile fragment for the Packages/McQ/Core directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/McQ/Core
SUBDIRS := \
	$(SRCDIR)/Algorithms \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/ThirdParty \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
