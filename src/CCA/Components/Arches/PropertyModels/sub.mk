# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/PropertyModels

SRCS += \
        $(SRCDIR)/PropertyModelBase.cc \
        $(SRCDIR)/PropertyModelFactory.cc \
        $(SRCDIR)/ExtentRxn.cc \
        $(SRCDIR)/TabStripFactor.cc \
        $(SRCDIR)/EmpSoot.cc \
        $(SRCDIR)/fvSootFromYsoot.cc \
        $(SRCDIR)/AlgebraicScalarDiss.cc \
        $(SRCDIR)/ScalarVarianceScaleSim.cc \
        $(SRCDIR)/HeatLoss.cc \
        $(SRCDIR)/ConstProperty.cc \
        $(SRCDIR)/NormScalarVariance.cc \
        $(SRCDIR)/RadProperties.cc \
        $(SRCDIR)/ScalarDissipation.cc 

