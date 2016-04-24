# Makefile fragment for this subdirectory

SRCDIR := CCA/Components/Arches/CoalModels

########################################################################
#
# CUDA_ENABLED_SRCS are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# WARNING: If you add a file to the list of CUDA_SRCS, you must add a
# corresponding rule at the end of this file!
#
# Also, do not put the .cc on this list of files as the .cc or .cu
# will be added automatically as needed.
#
CUDA_ENABLED_SRCS =              \
        PartVel                  \
        CharOxidation            \
        CharOxidationShaddix     \
        CharOxidationSmith       \
        ConstantModel            \
        Deposition               \
        Devolatilization         \
        DragModel                \
        EnthalpyShaddix          \
        FOWYDevol                \
        HeatTransfer             \
        KobayashiSarofimDevol    \
        MaximumTemperature       \
        ParticleConvection       \
        RichardsFletcherDevol    \
        SimpleBirth              \
        Thermophoresis           \
        YamamotoDevol            

ifeq ($(HAVE_CUDA),yes)
   # CUDA enabled files, listed here (and with a rule at the end of
   # this sub.mk) are copied to the binary side and renamed with a .cu
   # extension (.cc replaced with .cu) so that they can be compiled
   # using the nvcc compiler.
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(OBJTOP_ABS)/$(SRCDIR)/$(var).cu)
   DLINK_FILES := $(DLINK_FILES) $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).o)
else
   SRCS += $(foreach var,$(CUDA_ENABLED_SRCS),$(SRCDIR)/$(var).cc)
endif

########################################################################
# Normal source files:

SRCS += \
  $(SRCDIR)/CoalModelFactory.cc      \
  $(SRCDIR)/ModelBase.cc             

$(SRCDIR)/ShaddixHeatTransfer.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
$(SRCDIR)/EnthalpyShaddix.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/PartVel.cu : $(SRCTOP_ABS)/$(SRCDIR)/PartVel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CharOxidation.cu : $(SRCTOP_ABS)/$(SRCDIR)/CharOxidation.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CharOxidationShaddix.cu : $(SRCTOP_ABS)/$(SRCDIR)/CharOxidationShaddix.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CharOxidationSmith.cu : $(SRCTOP_ABS)/$(SRCDIR)/CharOxidationSmith.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ConstantModel.cu : $(SRCTOP_ABS)/$(SRCDIR)/ConstantModel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/Deposition.cu : $(SRCTOP_ABS)/$(SRCDIR)/Deposition.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/Devolatilization.cu : $(SRCTOP_ABS)/$(SRCDIR)/Devolatilization.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DragModel.cu : $(SRCTOP_ABS)/$(SRCDIR)/DragModel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/EnthalpyShaddix.cu : $(SRCTOP_ABS)/$(SRCDIR)/EnthalpyShaddix.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/FOWYDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/FOWYDevol.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/HeatTransfer.cu : $(SRCTOP_ABS)/$(SRCDIR)/HeatTransfer.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/KobayashiSarofimDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/KobayashiSarofimDevol.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/MaximumTemperature.cu : $(SRCTOP_ABS)/$(SRCDIR)/MaximumTemperature.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ParticleConvection.cu : $(SRCTOP_ABS)/$(SRCDIR)/ParticleConvection.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/RichardsFletcherDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/RichardsFletcherDevol.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/SimpleBirth.cu : $(SRCTOP_ABS)/$(SRCDIR)/SimpleBirth.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/Thermophoresis.cu : $(SRCTOP_ABS)/$(SRCDIR)/Thermophoresis.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/YamamotoDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/YamamotoDevol.cc
	cp $< $@
endif
