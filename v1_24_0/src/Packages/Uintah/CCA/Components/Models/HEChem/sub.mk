# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Models/HEChem

SRCS	+= \
       $(SRCDIR)/Simple_Burn.cc \
       $(SRCDIR)/Steady_Burn.cc \
       $(SRCDIR)/IandG.cc       \
       $(SRCDIR)/LightTime.cc   \
       $(SRCDIR)/JWLpp.cc
