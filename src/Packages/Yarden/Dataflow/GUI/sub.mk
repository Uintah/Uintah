# Makefile fragment for this subdirectory

SRCDIR := Packages/Yarden/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/TensorFieldReader.tcl \
	$(SRCDIR)/ViewTensors.tcl \
	$(SRCDIR)/TensorFieldWriter.tcl \
#[INSERT NEW TCL FILE HERE]
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Packages/Yarden/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

