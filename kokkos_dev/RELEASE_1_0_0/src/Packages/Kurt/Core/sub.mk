#Makefile fragment for the Packages/Kurt/Core directory

include $(SRCTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := Packages/Kurt/Core
SUBDIRS := \
	$(SRCDIR)/Datatypes

#	$(SRCDIR)/ThirdParty \
#	$(SRCDIR)/Algorithms \

include $(SRCTOP_ABS)/scripts/recurse.mk

PSELIBS := 
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP_ABS)/scripts/largeso_epilogue.mk
