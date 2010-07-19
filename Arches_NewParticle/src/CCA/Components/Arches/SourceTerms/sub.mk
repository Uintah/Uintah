# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/SourceTerms

SRCS += \
	$(SRCDIR)/SourceTermFactory.cc \
	$(SRCDIR)/SourceTermBase.cc \
	$(SRCDIR)/DevolMixtureFraction.cc \
	$(SRCDIR)/CharOxidationMixtureFraction.cc \
	$(SRCDIR)/ConstantSourceTerm.cc \
	$(SRCDIR)/WestbrookDryer.cc \
	$(SRCDIR)/MMS1.cc \
	$(SRCDIR)/ParticleGasMomentum.cc \

