# Makefile fragment for this subdirectory

SRCDIR := Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
			$(SRCDIR)/ArrowWidget.tcl \
			$(SRCDIR)/BaseWidget.tcl \
			$(SRCDIR)/Binop.tcl \
			$(SRCDIR)/BoxWidget.tcl \
			$(SRCDIR)/CriticalPointWidget.tcl \
			$(SRCDIR)/ComponentWizard.tcl \
			$(SRCDIR)/CrosshairWidget.tcl \
			$(SRCDIR)/Edge.tcl \
			$(SRCDIR)/FrameWidget.tcl \
			$(SRCDIR)/GaugeWidget.tcl \
			$(SRCDIR)/Gauss.tcl \
			$(SRCDIR)/ImageConvolve.tcl \
			$(SRCDIR)/LightWidget.tcl \
			$(SRCDIR)/MeshInterpVals.tcl \
			$(SRCDIR)/MeshToGeom.tcl \
			$(SRCDIR)/Module.tcl \
			$(SRCDIR)/NetworkEditor.tcl \
			$(SRCDIR)/Noise.tcl \
			$(SRCDIR)/PathWidget.tcl \
			$(SRCDIR)/PointWidget.tcl \
			$(SRCDIR)/Radon.tcl \
			$(SRCDIR)/RingWidget.tcl \
			$(SRCDIR)/ScaledBoxWidget.tcl \
			$(SRCDIR)/ScaledFrameWidget.tcl \
			$(SRCDIR)/Segment.tcl \
			$(SRCDIR)/Sharpen.tcl \
			$(SRCDIR)/Snakes.tcl \
			$(SRCDIR)/Subsample.tcl \
			$(SRCDIR)/Ted.tcl \
			$(SRCDIR)/Threshold.tcl \
			$(SRCDIR)/TiffReader.tcl \
			$(SRCDIR)/TiffWriter.tcl \
			$(SRCDIR)/Transforms.tcl \
			$(SRCDIR)/Unop.tcl\
			$(SRCDIR)/ViewWidget.tcl \
			$(SRCDIR)/defaults.tcl \
			$(SRCDIR)/devices.tcl \
			$(SRCDIR)/platformSpecific.tcl \
			$(SRCDIR)/ReadField.tcl \
			$(SRCDIR)/WriteField.tcl \
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Dataflow/GUI

#			$(SRCDIR)/Delaunay.tcl \
#			$(SRCDIR)/MeshView.tcl \


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

