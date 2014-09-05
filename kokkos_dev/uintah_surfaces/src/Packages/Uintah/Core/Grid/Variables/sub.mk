# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/Core/Grid/Variables

SRCS     += \
	$(SRCDIR)/CellIterator.cc $(SRCDIR)/CCVariableBase.cc \
	$(SRCDIR)/SFCXVariableBase.cc $(SRCDIR)/SFCYVariableBase.cc \
	$(SRCDIR)/SFCZVariableBase.cc \
	$(SRCDIR)/LocallyComputedPatchVarMap.cc \
	$(SRCDIR)/NCVariableBase.cc $(SRCDIR)/ParticleSet.cc \
	$(SRCDIR)/ParticleSubset.cc $(SRCDIR)/ParticleVariableBase.cc \
	$(SRCDIR)/ReductionVariableBase.cc \
	$(SRCDIR)/ReductionVariable_special.cc \
	$(SRCDIR)/SoleVariableBase.cc \
	$(SRCDIR)/SoleVariable_special.cc \
	$(SRCDIR)/Stencil7.cc \
	$(SRCDIR)/VarLabel.cc \
	$(SRCDIR)/VarLabelLevelDW.cc \
	$(SRCDIR)/VarLabelMatlPatch.cc \
	$(SRCDIR)/PSPatchMatlGhost.cc \
	$(SRCDIR)/Variable.cc \
	$(SRCDIR)/PerPatchBase.cc \
	$(SRCDIR)/ParticleVariable_special.cc \
	$(SRCDIR)/ComputeSet.cc \
	$(SRCDIR)/ComputeSet_special.cc \
	$(SRCDIR)/ugc_templates.cc 

#
#         $(SRCDIR)/ParticleVariable_templates.cc \
#         $(SRCDIR)/NCVariable_templates.cc \
#         $(SRCDIR)/CCVariable_templates.cc \
#         $(SRCDIR)/SFCXVariable_templates.cc \
#         $(SRCDIR)/SFCYVariable_templates.cc \
#         $(SRCDIR)/SFCZVariable_templates.cc
