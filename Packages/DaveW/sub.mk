#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/largeso_prologue.mk

SRCDIR := DaveW

SUBDIRS := $(SRCDIR)/Datatypes $(SRCDIR)/Modules $(SRCDIR)/ThirdParty \
	$(SRCDIR)/GUI

include $(OBJTOP_ABS)/scripts/recurse.mk

PSELIBS := PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/largeso_epilogue.mk

#  This must be done *after* the library block
SUBDIRS := $(SRCDIR)/convert 

include $(OBJTOP_ABS)/scripts/recurse.mk


#
# $Log$
# Revision 1.1  2000/03/17 09:25:15  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
