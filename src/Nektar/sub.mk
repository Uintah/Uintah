#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Nektar

SUBDIRS := $(SRCDIR)/GUI $(SRCDIR)/Datatypes $(SRCDIR)/Modules  

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := PSECommon PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

