# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/MIT/Dataflow/GUI

SRCS := \
	$(SRCDIR)/BayerAnalysis.tcl\
	$(SRCDIR)/DistributionReader.tcl\
	$(SRCDIR)/ItPDSimPartGui.tcl\
	$(SRCDIR)/RtPDSimPartGui.tcl\
	$(SRCDIR)/RUPDSimPartGui.tcl\
	$(SRCDIR)/IUniformPDSimPartGui.tcl\
	$(SRCDIR)/IGaussianPDSimPartGui.tcl\
	$(SRCDIR)/MeasurementsReader.tcl\
	$(SRCDIR)/Metropolis.tcl\
	$(SRCDIR)/MetropolisReader.tcl\
	$(SRCDIR)/MetropolisWriter.tcl\
	$(SRCDIR)/PPexample.tcl\
        $(SRCDIR)/PPexampleGui.tcl\
	$(SRCDIR)/Sampler.tcl\
	$(SRCDIR)/SamplerGui.tcl\
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk



