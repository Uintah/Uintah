# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Core/Variables

SRCS     += \
	$(SRCDIR)/CellIterator.cc $(SRCDIR)/CCVariableBase.cc \
	$(SRCDIR)/SFCXVariableBase.cc $(SRCDIR)/SFCYVariableBase.cc \
	$(SRCDIR)/SFCZVariableBase.cc \
	$(SRCDIR)/LocallyComputedPatchVarMap.cc \
	$(SRCDIR)/NCVariableBase.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ParticleVariableBase.cc \
	$(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/ReductionVariable_special.cc \
	$(SRCDIR)/Stencil7.cc \
	$(SRCDIR)/VarLabel.cc \
	$(SRCDIR)/VarLabelLevelDW.cc \
	$(SRCDIR)/VarLabelMatlPatch.cc \
	$(SRCDIR)/VarLabelMatlPatchDW.cc \
	$(SRCDIR)/PSPatchMatlGhost.cc \
	$(SRCDIR)/Variable.cc \
	$(SRCDIR)/PerPatchBase.cc \
	$(SRCDIR)/ParticleVariable_special.cc \
	$(SRCDIR)/ComputeSet.cc \
	$(SRCDIR)/ComputeSet_special.cc \
	$(SRCDIR)/ParticleVariable_templates.cc \
	$(SRCDIR)/NCVariable_templates.cc \
	$(SRCDIR)/CCVariable_templates.cc \
	$(SRCDIR)/SFCXVariable_templates.cc \
	$(SRCDIR)/SFCYVariable_templates.cc \
	$(SRCDIR)/SFCZVariable_templates.cc 

PSELIBS := \
	Packages/Uintah/Core/Math \

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

