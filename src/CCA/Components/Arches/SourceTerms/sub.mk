# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/SourceTerms

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
         $(SRCDIR)/CoalGasDevol.cc             \
         $(SRCDIR)/CoalGasDevolMom.cc          \
         $(SRCDIR)/CoalGasHeat.cc              \
         $(SRCDIR)/CoalGasMomentum.cc          \
         $(SRCDIR)/CoalGasOxi.cc               \
         $(SRCDIR)/CoalGasOxiMom.cc            \
         $(SRCDIR)/ConductiveHT.cc             \
         $(SRCDIR)/ZZNoxSolid.cc               \
         $(SRCDIR)/psNox.cc                    \
         $(SRCDIR)/DORadiation.cc              \
         $(SRCDIR)/MomentumDragSrc.cc          \
         $(SRCDIR)/RMCRT.cc                    \
         $(SRCDIR)/SourceTermFactory.cc        \
         $(SRCDIR)/UnweightedSrcTerm.cc        \
         $(SRCDIR)/BowmanNOx.cc                \
         $(SRCDIR)/BrownSoot.cc                \
         $(SRCDIR)/MoMICSoot.cc                \
				 $(SRCDIR)/MonoSoot.cc								 \
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
         $(SRCDIR)/SourceTermBase.cc           \
         $(SRCDIR)/TabRxnRate.cc               \
         $(SRCDIR)/WestbrookDryer.cc

########################################################################
#
# Automatically create rules to copy CUDA enabled source (.cc) files
# to the binary build tree and rename with a .cu extension.
#

ifeq ($(HAVE_CUDA),yes)

  # Call the make-cuda-target function on each of the CUDA_ENABLED_SRCS:
  $(foreach file,$(CUDA_ENABLED_SRCS),$(eval $(call make-cuda-target,$(file))))

endif
