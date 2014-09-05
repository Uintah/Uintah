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
	$(SRCDIR)/VarLabelMatlPatch.cc \
	$(SRCDIR)/Variable.cc \
	$(SRCDIR)/PerPatchBase.cc \
	$(SRCDIR)/GeometryPiece.cc \
	$(SRCDIR)/SphereGeometryPiece.cc \
	$(SRCDIR)/SphereMembraneGeometryPiece.cc \
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
	$(SRCDIR)/ParticleVariable_special.cc \
	$(SRCDIR)/fillFace_special.cc \
	$(SRCDIR)/ComputeSet_special.cc \
	$(SRCDIR)/ucg_templates.cc



PSELIBS := \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Util                        \
	Dataflow/XMLUtil

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) -lm $(GZ_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

