#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Util

SRCS     += $(SRCDIR)/ArrayTemplates.cc \
	$(SRCDIR)/BoundedArrayTemplates.cc \
	$(SRCDIR)/Matrix3.cc \
	$(SRCDIR)/MatrixTemplates.cc


#
# $Log$
# Revision 1.2  2000/05/26 22:27:25  tan
# Template files are not included.
#
# Revision 1.1  2000/03/17 09:29:41  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
