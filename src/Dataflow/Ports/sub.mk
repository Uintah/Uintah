# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Ports

SRCS     += $(SRCDIR)/ColorMapPort.cc             \
            $(SRCDIR)/ColumnMatrixPort.cc         \
            $(SRCDIR)/FieldPort.cc                \
            $(SRCDIR)/FieldWrapperPort.cc         \
            $(SRCDIR)/GeomPort.cc                 \
            $(SRCDIR)/GeometryPort.cc             \
	    $(SRCDIR)/GLTexture3DPort.cc          \
            $(SRCDIR)/HexMeshPort.cc              \
	    $(SRCDIR)/ImagePort.cc		  \
            $(SRCDIR)/MatrixPort.cc               \
            $(SRCDIR)/MeshPort.cc                 \
            $(SRCDIR)/PathPort.cc                 \
            $(SRCDIR)/ScalarFieldPort.cc          \
            $(SRCDIR)/SurfacePort.cc              \
            $(SRCDIR)/VectorFieldPort.cc          \
            $(SRCDIR)/VoidStarPort.cc             \
            $(SRCDIR)/templates.cc


PSELIBS := Dataflow/Network Dataflow/Comm Core/Containers \
	Core/Thread Core/Geom Core/Geometry Core/Exceptions \
	Core/Persistent Core/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

