#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Remote/Tools/Util

SRCS     += \
	$(SRCDIR)/Timer.cc \
	$(SRCDIR)/Utils.cc

#
# $Log$
# Revision 1.1  2000/07/10 20:44:24  dahart
# initial commit
#
# Revision 1.1  2000/06/06 15:29:31  dahart
# - Added a new package / directory tree for the remote visualization
# framework.
# - Added a new Salmon-derived module with a small server-daemon,
# network support, and a couple of very basic remote vis. services.
#
# Revision 1.1  2000/03/17 09:27:18  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
