#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Remote/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/remoteSalmon.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Remote/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.2  2000/06/06 16:39:10  dahart
# Fixed a bug
#
# Revision 1.1  2000/06/06 15:29:24  dahart
# - Added a new package / directory tree for the remote visualization
# framework.
# - Added a new Salmon-derived module with a small server-daemon,
# network support, and a couple of very basic remote vis. services.
#
# Revision 1.2  2000/03/20 19:36:50  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:45  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
