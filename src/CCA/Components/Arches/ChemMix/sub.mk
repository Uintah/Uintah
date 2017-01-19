# Makefile fragment for this subdirectory
 
SRCDIR := CCA/Components/Arches/ChemMix

#
# CUDA_ENABLED_SRCS are files that if CUDA is enabled (via configure), must be
# compiled using the nvcc compiler.
#
# Do not put the .cc on the file name as the .cc or .cu will be added automatically
# as needed.
#
CUDA_ENABLED_SRCS :=          

#ifeq ($(HAVE_TABPROPS),yes)
#   CUDA_ENABLED_SRCS += TabPropsInterface
#endif

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
      $(SRCDIR)/TabPropsInterface.cc      \
      $(SRCDIR)/TableLookup.cc            \
      $(SRCDIR)/ClassicTableInterface.cc  \
      $(SRCDIR)/ColdFlow.cc               \
      $(SRCDIR)/MixingRxnModel.cc         \
      $(SRCDIR)/ConstantProps.cc         

PSELIBS := $(PSELIBS) Core/IO

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)

  # Call the make-cuda-target function on each of the CUDA_ENABLED_SRCS:
  $(foreach file,$(CUDA_ENABLED_SRCS),$(eval $(call make-cuda-target,$(file))))

endif
