# Makefile fragment for this subdirectory

SRCDIR := Packages/Yarden/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/Isosurface.tcl \
	$(SRCDIR)/IsoSurfaceSAGE.tcl \
	$(SRCDIR)/IsoSurfaceNOISE.tcl \
	$(SRCDIR)/SearchNOISE.tcl \
	$(SRCDIR)/Span.tcl \
	$(SRCDIR)/TensorFieldReader.tcl \
	$(SRCDIR)/TensorFieldWriter.tcl \
	$(SRCDIR)/ViewTensors.tcl \
#[INSERT NEW TCL FILE HERE]
	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/Yarden/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

