# Makefile fragment for this subdirectory

SRCDIR := Packages/Yarden/Dataflow/GUI

SRCS := \
	$(SRCDIR)/Isosurface.tcl \
	$(SRCDIR)/IsoSurfaceSAGE.tcl \
	$(SRCDIR)/IsoSurfaceNOISE.tcl \
	$(SRCDIR)/SearchNOISE.tcl \
	$(SRCDIR)/Span.tcl \
	$(SRCDIR)/TensorFieldReader.tcl \
	$(SRCDIR)/TensorFieldWriter.tcl \
	$(SRCDIR)/ViewTensors.tcl \
#[INSERT NEW TCL FILE HERE]

include $(SCIRUN_SCRIPTS)/tclIndex.mk


