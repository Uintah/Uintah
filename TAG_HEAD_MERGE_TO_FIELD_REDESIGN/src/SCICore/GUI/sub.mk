#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := SCICore/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/BaseDial.tcl $(SRCDIR)/ColorPicker.tcl \
	$(SRCDIR)/DebugSettings.tcl $(SRCDIR)/Dial.tcl \
	$(SRCDIR)/Dialbox.tcl $(SRCDIR)/Doublefile.tcl \
	$(SRCDIR)/Filebox.tcl $(SRCDIR)/HelpPage.tcl \
	$(SRCDIR)/BioPSEFilebox.tcl\
	$(SRCDIR)/Histogram.tcl $(SRCDIR)/MaterialEditor.tcl \
	$(SRCDIR)/MemStats.tcl $(SRCDIR)/PointVector.tcl \
	$(SRCDIR)/ThreadStats.tcl $(SRCDIR)/Util.tcl \
	$(SRCDIR)/VTRDial.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/SCICore/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.4  2000/10/24 05:57:49  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/10/20 05:27:43  samsonov
# Initial commit: new file dialog
#
# Revision 1.2  2000/03/20 19:37:38  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:22  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
