#
# Makefile fragment for this subdirectory
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

