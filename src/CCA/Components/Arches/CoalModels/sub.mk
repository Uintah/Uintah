# Makefile fragment for this subdirectory

SRCDIR := CCA/Components/Arches/CoalModels

########################################################################
#
# CUDA_ENABLED_SRCS are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# Do not put the .cc on the file name as the .cc or .cu will be added automatically
# as needed.
#
CUDA_ENABLED_SRCS :=

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
        $(SRCDIR)/CharOxidation.cc            \
        $(SRCDIR)/CharOxidationShaddix.cc     \
        $(SRCDIR)/CharOxidationSmith.cc       \
        $(SRCDIR)/CharOxidationSmithConstLv0.cc       \
        $(SRCDIR)/ConstantModel.cc            \
        $(SRCDIR)/Deposition.cc               \
        $(SRCDIR)/Devolatilization.cc         \
        $(SRCDIR)/DragModel.cc                \
        $(SRCDIR)/EnthalpyShaddix.cc          \
        $(SRCDIR)/FOWYDevol.cc                \
        $(SRCDIR)/HeatTransfer.cc             \
        $(SRCDIR)/KobayashiSarofimDevol.cc    \
        $(SRCDIR)/MaximumTemperature.cc       \
        $(SRCDIR)/PartVel.cc                  \
        $(SRCDIR)/RichardsFletcherDevol.cc    \
        $(SRCDIR)/BirthDeath.cc               \
        $(SRCDIR)/Thermophoresis.cc           \
        $(SRCDIR)/LinearSwelling.cc           \
      	$(SRCDIR)/ShrinkageRate.cc            \
        $(SRCDIR)/YamamotoDevol.cc            \
        $(SRCDIR)/CoalModelFactory.cc         \
        $(SRCDIR)/ModelBase.cc

$(SRCDIR)/ShaddixHeatTransfer.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h
$(SRCDIR)/EnthalpyShaddix.$(OBJEXT): $(SRCDIR)/fortran/rqpart_fort.h

########################################################################
#
# Automatically create rules to copy CUDA enabled source (.cc) files
# to the binary build tree and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)

  # Call the make-cuda-target function on each of the CUDA_ENABLED_SRCS:
  $(foreach file,$(CUDA_ENABLED_SRCS),$(eval $(call make-cuda-target,$(file))))

endif
