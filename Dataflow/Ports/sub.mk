#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/Datatypes

SRCS     += $(SRCDIR)/AttribPort.cc               \
            $(SRCDIR)/BooleanPort.cc              \
            $(SRCDIR)/CameraViewPort.cc           \
            $(SRCDIR)/ColorMapPort.cc             \
            $(SRCDIR)/ColumnMatrixPort.cc         \
            $(SRCDIR)/DomainPort.cc               \
            $(SRCDIR)/FieldPort.cc                \
            $(SRCDIR)/FieldWrapperPort.cc         \
            $(SRCDIR)/GeomPort.cc                 \
            $(SRCDIR)/GeometryPort.cc             \
	    $(SRCDIR)/GLTexture3DPort.cc          \
            $(SRCDIR)/HexMeshPort.cc              \
            $(SRCDIR)/IntervalPort.cc             \
            $(SRCDIR)/MatrixPort.cc               \
            $(SRCDIR)/MeshPort.cc                 \
            $(SRCDIR)/PathPort.cc                 \
            $(SRCDIR)/ScalarFieldPort.cc          \
            $(SRCDIR)/ScaledBoxWidgetData.cc      \
            $(SRCDIR)/ScaledBoxWidgetDataPort.cc  \
            $(SRCDIR)/SoundPort.cc                \
            $(SRCDIR)/SpanPort.cc                 \
            $(SRCDIR)/SpanSpace.cc                \
            $(SRCDIR)/SpanTree.cc                 \
            $(SRCDIR)/SurfacePort.cc              \
            $(SRCDIR)/VectorFieldPort.cc          \
            $(SRCDIR)/VoidStarPort.cc             \
            $(SRCDIR)/cMatrixPort.cc              \
            $(SRCDIR)/cVectorPort.cc              \
            $(SRCDIR)/templates.cc


PSELIBS := PSECore/Dataflow PSECore/Comm SCICore/Containers \
	SCICore/Thread SCICore/Geom SCICore/Geometry SCICore/Exceptions \
	SCICore/Persistent SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

