# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Grid

SRCS     += \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/BufferInfo.cc $(SRCDIR)/PackBufferInfo.cc \
	$(SRCDIR)/CellIterator.cc $(SRCDIR)/CCVariableBase.cc \
	$(SRCDIR)/SFCXVariableBase.cc $(SRCDIR)/SFCYVariableBase.cc \
	$(SRCDIR)/SFCZVariableBase.cc \
	$(SRCDIR)/Grid.cc \
	$(SRCDIR)/Level.cc $(SRCDIR)/Material.cc \
	$(SRCDIR)/PatchRangeTree.cc \
	$(SRCDIR)/NCVariableBase.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ParticleVariableBase.cc \
	$(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/ReductionVariable_special.cc \
	$(SRCDIR)/Patch.cc \
	$(SRCDIR)/Ghost.cc \
	$(SRCDIR)/SimulationState.cc \
	$(SRCDIR)/SimulationTime.cc \
	$(SRCDIR)/Task.cc $(SRCDIR)/VarLabel.cc \
	$(SRCDIR)/Variable.cc \
	$(SRCDIR)/templates.cc $(SRCDIR)/PerPatchBase.cc \
	$(SRCDIR)/GeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc \
	$(SRCDIR)/BoxGeometryPiece.cc \
	$(SRCDIR)/CylinderGeometryPiece.cc \
	$(SRCDIR)/TriGeometryPiece.cc \
	$(SRCDIR)/UnionGeometryPiece.cc \
	$(SRCDIR)/DifferenceGeometryPiece.cc \
	$(SRCDIR)/IntersectionGeometryPiece.cc \
	$(SRCDIR)/GeometryPieceFactory.cc \
	$(SRCDIR)/VelocityBoundCond.cc \
	$(SRCDIR)/TemperatureBoundCond.cc \
	$(SRCDIR)/PressureBoundCond.cc \
	$(SRCDIR)/DensityBoundCond.cc \
	$(SRCDIR)/BoundCondFactory.cc \
	$(SRCDIR)/BoundCondData.cc \
	$(SRCDIR)/UnknownVariable.cc \
	$(SRCDIR)/SimpleMaterial.cc \
	$(SRCDIR)/NCVariable_special.cc \
	$(SRCDIR)/CCVariable_special.cc \
	$(SRCDIR)/SFCXVariable_special.cc \
	$(SRCDIR)/SFCYVariable_special.cc \
	$(SRCDIR)/SFCZVariable_special.cc \
	$(SRCDIR)/ucg_templates.cc




PSELIBS := \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) -lm -lz

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

