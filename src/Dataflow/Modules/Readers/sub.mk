# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Readers

SRCS     += \
	$(SRCDIR)/ColorMapReader.cc\
	$(SRCDIR)/ColumnMatrixReader.cc\
	$(SRCDIR)/DukeRawRead.cc\
	$(SRCDIR)/GeomReader.cc\
	$(SRCDIR)/GeometryReader.cc\
	$(SRCDIR)/ImageReader.cc\
	$(SRCDIR)/MatrixReader.cc\
	$(SRCDIR)/MeshReader.cc\
	$(SRCDIR)/MultiSFRGReader.cc\
	$(SRCDIR)/PointsReader.cc\
	$(SRCDIR)/ScalarFieldReader.cc\
	$(SRCDIR)/SurfaceReader.cc\
	$(SRCDIR)/VectorFieldReader.cc\
	$(SRCDIR)/VoidStarReader.cc\
	$(SRCDIR)/PathReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Core/Datatypes Core/Datatypes \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/Containers Core/TclInterface Core/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk
