# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

SRCDIR := Packages/ModelCreation/Dataflow/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: \
	$(SRCDIR)/ComputeTensorArray.tcl\
	$(SRCDIR)/ComputeVectorArray.tcl\
	$(SRCDIR)/ComputeScalarArray.tcl\
	$(SRCDIR)/ArrayInfo.tcl\
	$(SRCDIR)/MatrixInfo.tcl\
	$(SRCDIR)/FieldInfo.tcl\
	$(SRCDIR)/ParameterList.tcl\
	$(SRCDIR)/ParameterListMatrix.tcl\
	$(SRCDIR)/ParameterListString.tcl\
	$(SRCDIR)/BundleReader.tcl\
	$(SRCDIR)/BundleWriter.tcl\
	$(SRCDIR)/ColorMapReader.tcl\
	$(SRCDIR)/ColorMapWriter.tcl\
	$(SRCDIR)/ColorMap2Reader.tcl\
	$(SRCDIR)/ColorMap2Writer.tcl\
	$(SRCDIR)/FieldReader.tcl\
	$(SRCDIR)/FieldWriter.tcl\
	$(SRCDIR)/MatrixReader.tcl\
	$(SRCDIR)/MatrixWriter.tcl\
	$(SRCDIR)/PathReader.tcl\
	$(SRCDIR)/PathWriter.tcl\
#[INSERT NEW TCL FILE HERE]

	$(OBJTOP)/createTclIndex $(SRCTOP)/Packages/ModelCreation/Dataflow/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex


