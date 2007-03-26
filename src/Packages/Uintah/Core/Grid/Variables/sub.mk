# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/Core/Grid/Variables

SRCS     += \
	$(SRCDIR)/CCVariableBase.cc             \
	$(SRCDIR)/CellIterator.cc               \
	$(SRCDIR)/ComputeSet.cc                 \
	$(SRCDIR)/ComputeSet_special.cc         \
	$(SRCDIR)/GridIterator.cc               \
	$(SRCDIR)/LocallyComputedPatchVarMap.cc \
	$(SRCDIR)/NCVariableBase.cc             \
	$(SRCDIR)/ParticleSet.cc                \
	$(SRCDIR)/ParticleSubset.cc             \
	$(SRCDIR)/ParticleVariableBase.cc       \
	$(SRCDIR)/ParticleVariable_special.cc   \
	$(SRCDIR)/PerPatchBase.cc               \
	$(SRCDIR)/PSPatchMatlGhost.cc           \
	$(SRCDIR)/ReductionVariableBase.cc      \
	$(SRCDIR)/ReductionVariable_special.cc  \
	$(SRCDIR)/ScrubItem.cc                  \
	$(SRCDIR)/SFCXVariableBase.cc           \
	$(SRCDIR)/SFCYVariableBase.cc           \
	$(SRCDIR)/SFCZVariableBase.cc           \
	$(SRCDIR)/SoleVariableBase.cc           \
	$(SRCDIR)/SoleVariable_special.cc       \
	$(SRCDIR)/Stencil7.cc                   \
	$(SRCDIR)/ugc_templates.cc              \
	$(SRCDIR)/VarLabel.cc                   \
	$(SRCDIR)/Variable.cc                   

#
#         $(SRCDIR)/ParticleVariable_templates.cc \
#         $(SRCDIR)/NCVariable_templates.cc \
#         $(SRCDIR)/CCVariable_templates.cc \
#         $(SRCDIR)/SFCXVariable_templates.cc \
#         $(SRCDIR)/SFCYVariable_templates.cc \
#         $(SRCDIR)/SFCZVariable_templates.cc
