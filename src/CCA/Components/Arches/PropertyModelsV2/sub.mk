# Makefile fragment for this subdirectory

SRCDIR := CCA/Components/Arches/PropertyModelsV2

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
        $(SRCDIR)/BurnsChriston.cc          \
        $(SRCDIR)/CCVel.cc                  \
        $(SRCDIR)/CO.cc                     \
        $(SRCDIR)/ConstantProperty.cc       \
        $(SRCDIR)/DensityPredictor.cc       \
        $(SRCDIR)/ContinuityPredictor.cc    \
        $(SRCDIR)/DensityStar.cc    \
        $(SRCDIR)/DensityRK.cc    \
        $(SRCDIR)/Drhodt.cc    \
        $(SRCDIR)/FaceVelocities.cc         \
        $(SRCDIR)/OneDWallHT.cc             \
        $(SRCDIR)/PropertyModelFactoryV2.cc \
        $(SRCDIR)/UFromRhoU.cc              \
        $(SRCDIR)/VariableStats.cc          \
        $(SRCDIR)/WallHFVariable.cc         \
        $(SRCDIR)/UnweightVariable.cc       \
        $(SRCDIR)/cloudBenchmark.cc         \
        $(SRCDIR)/gasRadProperties.cc       \
        $(SRCDIR)/partRadProperties.cc      \
        $(SRCDIR)/sootVolumeFrac.cc         \
        $(SRCDIR)/ConsScalarDiffusion.cc         \
        $(SRCDIR)/spectralProperties.cc     \
        $(SRCDIR)/GasKineticEnergy.cc     \
        $(SRCDIR)/sumRadiation.cc

########################################################################
#
# Automatically create rules to copy CUDA enabled source (.cc) files
# to the binary build tree and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)

  # Call the make-cuda-target function on each of the CUDA_ENABLED_SRCS:
  $(foreach file,$(CUDA_ENABLED_SRCS),$(eval $(call make-cuda-target,$(file))))

endif
