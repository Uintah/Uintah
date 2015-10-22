# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/ParticleModels

SRCS += \
        $(SRCDIR)/BodyForce.cc \
        $(SRCDIR)/Constant.cc \
        $(SRCDIR)/CQMOMSourceWrapper.cc \
        $(SRCDIR)/ExampleParticleModel.cc \
        $(SRCDIR)/DragModel.cc \
				$(SRCDIR)/CoalDensity.cc \
				$(SRCDIR)/CoalTemperature.cc \
				$(SRCDIR)/CoalTemperatureNebo.cc \
				$(SRCDIR)/CoalMassClip.cc \
				$(SRCDIR)/RateDeposition.cc \
        $(SRCDIR)/TotNumDensity.cc \
        $(SRCDIR)/ParticleModelFactory.cc \
        $(SRCDIR)/FOWYDevol.cc \
        $(SRCDIR)/ShaddixOxidation
