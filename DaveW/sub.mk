#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := DaveW

SUBDIRS := $(SRCDIR)/Datatypes $(SRCDIR)/Modules $(SRCDIR)/ThirdParty \
	$(SRCDIR)/GUI

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

#  This must be done *after* the library block
SUBDIRS := $(SRCDIR)/convert 

include $(SRCTOP)/scripts/recurse.mk


#
# $Log$
# Revision 1.2  2000/03/20 19:35:49  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:15  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
