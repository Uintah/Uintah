#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Yarden

SUBDIRS := \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/Modules \
	$(SRCDIR)/GUI \
	$(SRCDIR)/convert 

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

