# Makefile fragment for this subdirectory

SRCDIR := Core/GUI

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
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Core/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

