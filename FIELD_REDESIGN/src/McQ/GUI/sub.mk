#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := McQ/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/Euler.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/McQ/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.2  2000/03/20 19:36:47  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:41  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
