#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := SCICore/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/BaseDial.tcl $(SRCDIR)/ColorPicker.tcl \
	$(SRCDIR)/DebugSettings.tcl $(SRCDIR)/Dial.tcl \
	$(SRCDIR)/Dialbox.tcl $(SRCDIR)/Doublefile.tcl \
	$(SRCDIR)/Filebox.tcl $(SRCDIR)/HelpPage.tcl \
	$(SRCDIR)/Histogram.tcl $(SRCDIR)/MaterialEditor.tcl \
	$(SRCDIR)/MemStats.tcl $(SRCDIR)/PointVector.tcl \
	$(SRCDIR)/ThreadStats.tcl $(SRCDIR)/Util.tcl \
	$(SRCDIR)/VTRDial.tcl
	scripts/createTclIndex SCICore/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.1  2000/03/17 09:28:22  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
