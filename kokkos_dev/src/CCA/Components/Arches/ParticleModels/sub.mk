# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/ParticleModels

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
CUDA_ENABLED_SRCS =           \
         CoalMassClip         \
         CoalTemperatureNebo  \
         DepositionVelocity   \
         RateDeposition       \
         BodyForce            \
         CQMOMSourceWrapper   \
         CoalDensity          \
         CoalTemperature      \
         Constant             \
         DragModel            \
         ExampleParticleModel \
         FOWYDevol            \
         ParticleModelFactory \
         ShaddixOxidation     \
         TotNumDensity        

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

# SRCS += ???

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/CoalMassClip.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalMassClip.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalTemperatureNebo.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalTemperatureNebo.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DepositionVelocity.cu : $(SRCTOP_ABS)/$(SRCDIR)/DepositionVelocity.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/RateDeposition.cu : $(SRCTOP_ABS)/$(SRCDIR)/RateDeposition.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/BodyForce.cu : $(SRCTOP_ABS)/$(SRCDIR)/BodyForce.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CQMOMSourceWrapper.cu : $(SRCTOP_ABS)/$(SRCDIR)/CQMOMSourceWrapper.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalDensity.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalDensity.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalTemperature.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalTemperature.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/Constant.cu : $(SRCTOP_ABS)/$(SRCDIR)/Constant.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DragModel.cu : $(SRCTOP_ABS)/$(SRCDIR)/DragModel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ExampleParticleModel.cu : $(SRCTOP_ABS)/$(SRCDIR)/ExampleParticleModel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/FOWYDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/FOWYDevol.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ParticleModelFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/ParticleModelFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ShaddixOxidation.cu : $(SRCTOP_ABS)/$(SRCDIR)/ShaddixOxidation.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TotNumDensity.cu : $(SRCTOP_ABS)/$(SRCDIR)/TotNumDensity.cc
	cp $< $@
endif
