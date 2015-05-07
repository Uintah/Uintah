# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/Transport

SRCS += \
        $(SRCDIR)/ScalarRHS.cc \
				$(SRCDIR)/URHS.cc \
				$(SRCDIR)/FEUpdate.cc \
				$(SRCDIR)/SSPInt.cc \
				$(SRCDIR)/TransportFactory.cc 
