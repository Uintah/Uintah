# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/TransportEqns

SRCS += \
  $(SRCDIR)/Discretization_new.cc \
  $(SRCDIR)/DQMOMEqn.cc           \
  $(SRCDIR)/DQMOMEqnFactory.cc    \
  $(SRCDIR)/EqnBase.cc            \
  $(SRCDIR)/EqnFactory.cc         \
  $(SRCDIR)/ScalarEqn.cc          
