# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/PropertyModelsV2

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
CUDA_ENABLED_SRCS =            \
        CO                     \
        DensityPredictor       \
        VariableStats          \
        WallHFVariable         \
        PropertyModelFactoryV2 

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

  $(OBJTOP_ABS)/$(SRCDIR)/CO.cu : $(SRCTOP_ABS)/$(SRCDIR)/CO.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DensityPredictor.cu : $(SRCTOP_ABS)/$(SRCDIR)/DensityPredictor.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/VariableStats.cu : $(SRCTOP_ABS)/$(SRCDIR)/VariableStats.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/WallHFVariable.cu : $(SRCTOP_ABS)/$(SRCDIR)/WallHFVariable.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/PropertyModelFactoryV2.cu : $(SRCTOP_ABS)/$(SRCDIR)/PropertyModelFactoryV2.cc
	cp $< $@
endif
