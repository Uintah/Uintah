# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/Transport

SRCS += \
        $(SRCDIR)/ScalarRHS.cc \
        $(SRCDIR)/KScalarRHS.cc \
				$(SRCDIR)/URHS.cc \
				$(SRCDIR)/FEUpdate.cc \
				$(SRCDIR)/KFEUpdate.cc \
				$(SRCDIR)/SSPInt.cc \
				$(SRCDIR)/TransportFactory.cc
