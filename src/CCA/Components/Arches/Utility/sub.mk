# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/Utility

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
CUDA_ENABLED_SRCS =                      \
          GridInfo                       \
          InitializeFactory              \
          InitLagrangianParticleSize     \
          InitLagrangianParticleVelocity \
          RandParticleLoc                \
          SurfaceNormals                 \
          TaskAlgebra                    \
          UtilityFactory                 \
          WaveFormInit                   

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
# Add normal source files:

# SRCS += ???

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/GridInfo.cu : $(SRCTOP_ABS)/$(SRCDIR)/GridInfo.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/InitializeFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/InitializeFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/InitLagrangianParticleSize.cu : $(SRCTOP_ABS)/$(SRCDIR)/InitLagrangianParticleSize.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/InitLagrangianParticleVelocity.cu : $(SRCTOP_ABS)/$(SRCDIR)/InitLagrangianParticleVelocity.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/RandParticleLoc.cu : $(SRCTOP_ABS)/$(SRCDIR)/RandParticleLoc.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/SurfaceNormals.cu : $(SRCTOP_ABS)/$(SRCDIR)/SurfaceNormals.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TaskAlgebra.cu : $(SRCTOP_ABS)/$(SRCDIR)/TaskAlgebra.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/UtilityFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/UtilityFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/WaveFormInit.cu : $(SRCTOP_ABS)/$(SRCDIR)/WaveFormInit.cc
	cp $< $@

endif
