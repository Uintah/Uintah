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

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Metropolis.tcl\
	$(SRCDIR)/Sampler.tcl\
	$(SRCDIR)/SamplerGui.tcl\
	$(SRCDIR)/BayerAnalysis.tcl\
	$(SRCDIR)/MeasurementsReader.tcl\
	$(SRCDIR)/DistributionReader.tcl\
	$(SRCDIR)/MetropolisWriter.tcl\
	$(SRCDIR)/MetropolisReader.tcl\
	$(SRCDIR)/PPexample.tcl\
        $(SRCDIR)/PPexampleGui.tcl\
	$(SRCDIR)/ItPDSimPartGui.tcl\
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/MIT/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


