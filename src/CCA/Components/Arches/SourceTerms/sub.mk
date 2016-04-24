# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/SourceTerms

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
CUDA_ENABLED_SRCS =               \
         CoalGasDevol             \
         CoalGasDevolMom          \
         CoalGasHeat              \
         CoalGasMomentum          \
         CoalGasOxi               \
         CoalGasOxiMom            \
         DORadiation              \
         MomentumDragSrc          \
         RMCRT                    \
         SourceTermFactory        \
         UnweightedSrcTerm        


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
         $(SRCDIR)/BowmanNOx.cc                \
         $(SRCDIR)/BrownSootFormation_Tar.cc   \
         $(SRCDIR)/BrownSootFormation_nd.cc    \
         $(SRCDIR)/BrownSootFormation_rhoYs.cc \
         $(SRCDIR)/ConstSrcTerm.cc             \
         $(SRCDIR)/DissipationSource.cc        \
         $(SRCDIR)/Inject.cc                   \
         $(SRCDIR)/IntrusionInlet.cc           \
         $(SRCDIR)/MMS1.cc                     \
         $(SRCDIR)/ManifoldRxn.cc              \
         $(SRCDIR)/PCTransport.cc              \
         $(SRCDIR)/SecondMFMoment.cc           \
         $(SRCDIR)/ShunnMoinMMSCont.cc         \
         $(SRCDIR)/ShunnMoinMMSMF.cc           \
         $(SRCDIR)/SootMassBalance.cc          \
         $(SRCDIR)/SourceTermBase.cc           \
         $(SRCDIR)/TabRxnRate.cc               \
         $(SRCDIR)/WestbrookDryer.cc           

########################################################################
#
# Rules to copy CUDA enabled source (.cc) files to the binary build tree
# and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)
  # Copy the 'original' .cc files into the binary tree and rename as .cu

  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasDevol.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasDevol.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasDevolMom.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasDevolMom.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasHeat.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasHeat.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasMomentum.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasMomentum.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasOxi.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasOxi.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/CoalGasOxiMom.cu : $(SRCTOP_ABS)/$(SRCDIR)/CoalGasOxiMom.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/DORadiation.cu : $(SRCTOP_ABS)/$(SRCDIR)/DORadiation.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/MomentumDragSrc.cu : $(SRCTOP_ABS)/$(SRCDIR)/MomentumDragSrc.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/RMCRT.cu : $(SRCTOP_ABS)/$(SRCDIR)/RMCRT.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/SourceTermFactory.cu : $(SRCTOP_ABS)/$(SRCDIR)/SourceTermFactory.cc
	cp $< $@
  $(OBJTOP_ABS)/$(SRCDIR)/UnweightedSrcTerm.cu : $(SRCTOP_ABS)/$(SRCDIR)/UnweightedSrcTerm.cc
	cp $< $@
endif
