#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/Util

SRCS     += $(SRCDIR)/Array.cc $(SRCDIR)/ArrayTemplates.cc \
	$(SRCDIR)/BoundedArray.cc $(SRCDIR)/BoundedArrayTemplates.cc \
	$(SRCDIR)/Matrix.cc $(SRCDIR)/Matrix3.cc \
	$(SRCDIR)/MatrixTemplates.cc $(SRCDIR)/SymmetricMatrix.cc


#
# $Log$
# Revision 1.1  2000/03/17 09:29:41  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
