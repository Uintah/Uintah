# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/application.tcl \
	$(SRCDIR)/test.tcl \
	$(SRCDIR)/ScalarFieldAverage.tcl \
	$(SRCDIR)/SubFieldHistogram.tcl \
	$(SRCDIR)/FieldExtractor.tcl \
	$(SRCDIR)/ScalarFieldExtractor.tcl $(SRCDIR)/TimestepSelector.tcl \
	$(SRCDIR)/VectorFieldExtractor.tcl \
	$(SRCDIR)/TensorFieldExtractor.tcl \
	$(SRCDIR)/ParticleFieldExtractor.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl $(SRCDIR)/ParticleVis.tcl \
	$(SRCDIR)/NodeHedgehog.tcl \
	$(SRCDIR)/ArchiveReader.tcl \
	$(SRCDIR)/GridVisualizer.tcl\
	$(SRCDIR)/PatchVisualizer.tcl\
	$(SRCDIR)/PatchDataVisualizer.tcl\
	$(SRCDIR)/RescaleColorMap.tcl\
	$(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/CuttingPlane.tcl\
	$(SRCDIR)/Hedgehog.tcl\
	$(SRCDIR)/ScalarOperator.tcl\
	$(SRCDIR)/ScalarFieldOperator.tcl\
	$(SRCDIR)/TensorOperator.tcl\
	$(SRCDIR)/TensorFieldOperator.tcl\
	$(SRCDIR)/TensorParticlesOperator.tcl\
	$(SRCDIR)/VectorFieldOperator.tcl\
	$(SRCDIR)/VectorParticlesOperator.tcl\
	$(SRCDIR)/AnimatedStreams.tcl\
	$(SRCDIR)/VariablePlotter.tcl\
#[INSERT NEW TCL FILE HERE]
#	$(SRCDIR)/EigenEvaluator.tcl\
#	$(SRCDIR)/ParticleEigenEvaluator.tcl\


	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Uintah/Dataflow/GUI


CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

