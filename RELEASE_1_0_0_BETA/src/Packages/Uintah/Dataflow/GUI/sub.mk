# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/application.tcl \
	$(SRCDIR)/test.tcl \
	$(SRCDIR)/ScalarFieldExtractor.tcl $(SRCDIR)/TimestepSelector.tcl \
	$(SRCDIR)/VectorFieldExtractor.tcl \
	$(SRCDIR)/TensorFieldExtractor.tcl \
	$(SRCDIR)/ParticleFieldExtractor.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl $(SRCDIR)/ParticleVis.tcl \
	$(SRCDIR)/NodeHedgehog.tcl $(SRCDIR)/ArchiveReader.tcl \
	$(SRCDIR)/GridVisualizer.tcl\
	$(SRCDIR)/EigenEvaluator.tcl\
	$(SRCDIR)/ParticleEigenEvaluator.tcl\
	$(SRCDIR)/TensorOperator.tcl\
	$(SRCDIR)/TensorFieldOperator.tcl\
	$(SRCDIR)/TensorParticlesOperator.tcl\
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Packages/Uintah/Dataflow/GUI


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

