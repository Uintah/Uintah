#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := SCIRun/Modules

SUBDIRS := \
	$(SRCDIR)/Image\
	 $(SRCDIR)/Mesh\
#[INSERT NEW SUBDIRS HERE]

include $(SRCTOP)/scripts/recurse.mk

#
# $Log$
# Revision 1.3  2000/06/07 17:32:58  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:38:10  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:00  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
