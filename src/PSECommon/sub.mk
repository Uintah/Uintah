#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := PSECommon

SUBDIRS := $(SRCDIR)/ThirdParty $(SRCDIR)/Algorithms $(SRCDIR)/Modules \
	 $(SRCDIR)/GUI

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := PSECore SCICore
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk


#
# $Log$
# Revision 1.3  2000/07/22 18:02:40  yarden
# add Algorithm subdir.
#
# Revision 1.2  2000/03/20 19:36:49  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
