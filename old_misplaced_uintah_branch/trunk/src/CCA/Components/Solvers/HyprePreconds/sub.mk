# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Solvers/HyprePreconds

SRCS += $(SRCDIR)/HyprePrecondBase.cc \
        $(SRCDIR)/HyprePrecondSMG.cc \
        $(SRCDIR)/HyprePrecondPFMG.cc \
        $(SRCDIR)/HyprePrecondSparseMSG.cc \
        $(SRCDIR)/HyprePrecondJacobi.cc \
        $(SRCDIR)/HyprePrecondDiagonal.cc
