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
	$(SRCDIR)/MaterialProperties.cc \
	$(SRCDIR)/PatchRangeTree.cc \
	$(SRCDIR)/NCVariableBase.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ParticleVariableBase.cc \
	$(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/ReductionVariable_special.cc \
	$(SRCDIR)/Patch.cc \
	$(SRCDIR)/Ghost.cc \
	$(SRCDIR)/SimulationState.cc \
	$(SRCDIR)/SimulationTime.cc \
	$(SRCDIR)/Stencil7.cc \
	$(SRCDIR)/Task.cc $(SRCDIR)/VarLabel.cc \
	$(SRCDIR)/VarLabelLevelDW.cc \
	$(SRCDIR)/VarLabelMatlPatch.cc \
	$(SRCDIR)/VarLabelMatlPatchDW.cc \
	$(SRCDIR)/Variable.cc \
	$(SRCDIR)/PerPatchBase.cc \
	$(SRCDIR)/VelocityBoundCond.cc \
	$(SRCDIR)/TemperatureBoundCond.cc \
	$(SRCDIR)/PressureBoundCond.cc \
	$(SRCDIR)/DensityBoundCond.cc \
	$(SRCDIR)/BoundCondFactory.cc \
	$(SRCDIR)/BoundCondReader.cc \
	$(SRCDIR)/BoundCondData.cc \
	$(SRCDIR)/BCDataArray.cc \
	$(SRCDIR)/UnionBCData.cc \
	$(SRCDIR)/DifferenceBCData.cc \
	$(SRCDIR)/SideBCData.cc \
	$(SRCDIR)/CircleBCData.cc \
	$(SRCDIR)/RectangleBCData.cc \
	$(SRCDIR)/UnknownVariable.cc \
	$(SRCDIR)/SimpleMaterial.cc \
	$(SRCDIR)/ParticleVariable_special.cc \
	$(SRCDIR)/fillFace_special.cc \
	$(SRCDIR)/ComputeSet_special.cc \
	$(SRCDIR)/ucg_templates.cc 

SUBDIRS := $(SRCDIR)/GeomPiece

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := \
	Packages/Uintah/Core/Math        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/ProblemSpec \
	Core/Thread                      \
	Core/Exceptions                  \
	Core/Geometry                    \
	Core/Containers                  \
	Core/Util

LIBS := $(XML_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

