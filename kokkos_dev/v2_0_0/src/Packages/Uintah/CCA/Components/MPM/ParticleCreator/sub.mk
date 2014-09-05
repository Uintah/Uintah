# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/MPM/ParticleCreator

SRCS     += \
	$(SRCDIR)/ParticleCreator.cc	\
	$(SRCDIR)/DefaultParticleCreator.cc	\
	$(SRCDIR)/ImplicitParticleCreator.cc	\
	$(SRCDIR)/FractureParticleCreator.cc	\
	$(SRCDIR)/MembraneParticleCreator.cc	\
	$(SRCDIR)/ShellParticleCreator.cc

PSELIBS := Packages/Uintah/Core/Grid \
	Packages/Uintah/CCA/Components/ICE \
	Packages/Uintah/CCA/Components/HETransformation \
	Core/Datatypes \
	Core/Util

