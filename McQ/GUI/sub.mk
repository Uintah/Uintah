#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := McQ/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Euler.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/McQ/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.3  2000/10/24 05:57:23  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.2  2000/03/20 19:36:47  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:41  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
