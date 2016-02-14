# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/TransportEqns

SRCS += \
  $(SRCDIR)/CQMOM_Convection_OpSplit.cc   \
  $(SRCDIR)/CQMOMEqn.cc           \
  $(SRCDIR)/CQMOMEqnFactory.cc    \
  $(SRCDIR)/CQMOM_Convection.cc   \
  $(SRCDIR)/Discretization_new.cc \
  $(SRCDIR)/DQMOMEqn.cc           \
  $(SRCDIR)/DQMOMEqnFactory.cc    \
  $(SRCDIR)/EqnBase.cc            \
  $(SRCDIR)/EqnFactory.cc         \
  $(SRCDIR)/ScalarEqn.cc          
