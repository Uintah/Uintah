# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/TransportEqns

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
CUDA_ENABLED_SRCS =         \
         CQMOMEqn           \
         CQMOMEqnFactory    \
         DQMOMEqn           \
         DQMOMEqnFactory    \
         EqnBase            \
         EqnFactory         \
         ScalarEqn                  

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

SRCS += \
  $(SRCDIR)/CQMOM_Convection.cc           \
  $(SRCDIR)/CQMOM_Convection_OpSplit.cc   \
  $(SRCDIR)/Discretization_new.cc         

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/CQMOMEqn.cu : $(SRCTOP_ABS)/$(SRCDIR)/CQMOMEqn.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CQMOMEqnFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/CQMOMEqnFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DQMOMEqn.cu : $(SRCTOP_ABS)/$(SRCDIR)/DQMOMEqn.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DQMOMEqnFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/DQMOMEqnFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/EqnBase.cu : $(SRCTOP_ABS)/$(SRCDIR)/EqnBase.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/EqnFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/EqnFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/ScalarEqn.cu : $(SRCTOP_ABS)/$(SRCDIR)/ScalarEqn.cc
	cp $< $@
endif
