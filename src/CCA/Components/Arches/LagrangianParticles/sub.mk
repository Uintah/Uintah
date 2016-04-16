# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/LagrangianParticles

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
CUDA_ENABLED_SRCS =                \
         LagrangianParticleFactory \
         UpdateParticlePosition    \
         UpdateParticleSize        \
         UpdateParticleVelocity

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

  $(OBJTOP_ABS)/$(SRCDIR)/LagrangianParticleFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/LagrangianParticleFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/UpdateParticlePosition.cu : $(SRCTOP_ABS)/$(SRCDIR)/UpdateParticlePosition.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/UpdateParticleSize.cu : $(SRCTOP_ABS)/$(SRCDIR)/UpdateParticleSize.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/UpdateParticleVelocity.cu : $(SRCTOP_ABS)/$(SRCDIR)/UpdateParticleVelocity.cc
	cp $< $@
endif
