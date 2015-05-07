# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/Arches/LagrangianParticles

SRCS += \
				$(SRCDIR)/UpdateParticlePosition.cc \
				$(SRCDIR)/UpdateParticleVelocity.cc \
				$(SRCDIR)/UpdateParticleSize.cc \
        $(SRCDIR)/LagrangianParticleFactory.cc 
