#Makefile fragment for the Packages/mmiller/Core directory

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/mmiller/Core
SUBDIRS := \
	$(SRCDIR)/Algorithms \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/ThirdParty \

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk
