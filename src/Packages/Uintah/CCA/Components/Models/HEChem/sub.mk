# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/Models/HEChem

SRCS	+= \
       $(SRCDIR)/Steady_Burn.cc \
       $(SRCDIR)/IandG.cc       \
       $(SRCDIR)/LightTime.cc   \
       $(SRCDIR)/JWLpp.cc
#       $(SRCDIR)/Simple_Burn.cc \
