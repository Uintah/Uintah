# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/ChemMix

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
CUDA_ENABLED_SRCS =          \
      ClassicTableInterface  \
      ColdFlow               \
      MixingRxnModel

ifeq ($(HAVE_TABPROPS),yes)
   CUDA_ENABLED_SRCS += TabPropsInterface
endif

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

SRCS += \
        $(SRCDIR)/ConstantProps.cc         

PSELIBS := $(PSELIBS) Core/IO

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # If Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/ClassicTableInterface.cu : $(SRCTOP_ABS)/$(SRCDIR)/ClassicTableInterface.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ColdFlow.cu : $(SRCTOP_ABS)/$(SRCDIR)/ColdFlow.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/MixingRxnModel.cu : $(SRCTOP_ABS)/$(SRCDIR)/MixingRxnModel.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/TabPropsInterface.cu : $(SRCTOP_ABS)/$(SRCDIR)/TabPropsInterface.cc
	cp $< $@

endif
