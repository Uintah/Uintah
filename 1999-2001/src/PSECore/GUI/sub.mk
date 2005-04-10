#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := PSECore/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ArrowWidget.tcl $(SRCDIR)/BaseWidget.tcl \
	$(SRCDIR)/BoxWidget.tcl $(SRCDIR)/CriticalPointWidget.tcl \
	$(SRCDIR)/CrosshairWidget.tcl $(SRCDIR)/FrameWidget.tcl \
	$(SRCDIR)/GaugeWidget.tcl $(SRCDIR)/LightWidget.tcl \
	$(SRCDIR)/Module.tcl $(SRCDIR)/NetworkEditor.tcl \
	$(SRCDIR)/PathWidget.tcl $(SRCDIR)/PointWidget.tcl \
	$(SRCDIR)/RingWidget.tcl $(SRCDIR)/ScaledBoxWidget.tcl \
	$(SRCDIR)/ScaledFrameWidget.tcl $(SRCDIR)/ViewWidget.tcl \
	$(SRCDIR)/defaults.tcl $(SRCDIR)/devices.tcl \
	$(SRCDIR)/platformSpecific.tcl $(SRCDIR)/ComponentWizard.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/PSECore/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.4  2000/10/24 05:57:42  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/10/15 04:34:29  moulding
# more of Phase 1 for new module maker
#
# Revision 1.2  2000/03/20 19:37:23  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:02  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
