# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/ColorMapReader.cc\
	$(SRCDIR)/ColorMapWriter.cc\
	$(SRCDIR)/ColumnMatrixReader.cc\
	$(SRCDIR)/ColumnMatrixWriter.cc\
	$(SRCDIR)/FieldReader.cc\
	$(SRCDIR)/FieldWriter.cc\
	$(SRCDIR)/GeomReader.cc\
	$(SRCDIR)/GeomWriter.cc\
	$(SRCDIR)/ImageReader.cc\
	$(SRCDIR)/MatrixReader.cc\
	$(SRCDIR)/MatrixWriter.cc\
	$(SRCDIR)/MeshReader.cc\
	$(SRCDIR)/MeshWriter.cc\
	$(SRCDIR)/PathReader.cc\
	$(SRCDIR)/PathWriter.cc\
	$(SRCDIR)/ScalarFieldReader.cc\
	$(SRCDIR)/ScalarFieldWriter.cc\
	$(SRCDIR)/SurfaceReader.cc\
	$(SRCDIR)/SurfaceWriter.cc\
	$(SRCDIR)/VectorFieldReader.cc\
	$(SRCDIR)/VectorFieldWriter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Core/Datatypes Core/Persistent \
	Core/Exceptions Core/Thread Core/Containers \
	Core/TclInterface Core/Geometry Core/Datatypes \
	Core/Util Core/Geom Core/TkExtensions \
	Dataflow/Widgets
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
