# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Core/Datatypes/General

SRCS     += $(SRCDIR)/ContourSet.cc $(SRCDIR)/ContourSetPort.cc \
	$(SRCDIR)/ManhattanDist.cc \
	$(SRCDIR)/ScalarTriSurfFieldace.cc \
	$(SRCDIR)/SegFld.cc $(SRCDIR)/SegFldPort.cc \
	$(SRCDIR)/SigmaSet.cc $(SRCDIR)/SigmaSetPort.cc \
        $(SRCDIR)/TopoSurfTree.cc \
#	$(SRCDIR)/VectorFieldMI.cc \
#	$(SRCDIR)/TensorField.cc $(SRCDIR)/TensorFieldBase.cc \
#	$(SRCDIR)/TensorFieldPort.cc \

PSELIBS := Dataflow/Network Core/Persistent Core/Geometry \
	Core/Exceptions Core/Datatypes Core/Thread \
	Core/Containers 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

