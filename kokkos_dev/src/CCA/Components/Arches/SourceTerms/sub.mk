# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/SourceTerms

SRCS += \
        $(SRCDIR)/SourceTermFactory.cc \
        $(SRCDIR)/SourceTermBase.cc \
        $(SRCDIR)/CoalGasDevol.cc \
        $(SRCDIR)/CoalGasOxi.cc \
        $(SRCDIR)/CoalGasHeat.cc \
        $(SRCDIR)/ConstSrcTerm.cc \
        $(SRCDIR)/WarchesSrc.cc \
        $(SRCDIR)/UnweightedSrcTerm.cc \
        $(SRCDIR)/WestbrookDryer.cc \
        $(SRCDIR)/CoalGasMomentum.cc \
        $(SRCDIR)/Inject.cc \
        $(SRCDIR)/TabRxnRate.cc \
        $(SRCDIR)/IntrusionInlet.cc \
        $(SRCDIR)/WasatchExprSource.cc \
	 			$(SRCDIR)/DORadiation.cc \
	 			$(SRCDIR)/RMCRT.cc \
	 			$(SRCDIR)/BowmanNOx.cc \
				$(SRCDIR)/PCTransport.cc \
        $(SRCDIR)/MMS1.cc \
        $(SRCDIR)/SecondMFMoment.cc \
				$(SRCDIR)/ManifoldRxn.cc \
        $(SRCDIR)/DissipationSource.cc \
        $(SRCDIR)/MomentumDragSrc.cc



