# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/CCA/Components

SUBDIRS := \
	$(SRCDIR)/DataArchiver \
	$(SRCDIR)/Schedulers \
	$(SRCDIR)/SimulationController \
	$(SRCDIR)/MPM \
	$(SRCDIR)/ICE \
	$(SRCDIR)/MPMICE \
	$(SRCDIR)/MPMArches \
	$(SRCDIR)/Arches \
	$(SRCDIR)/ProblemSpecification

include $(SRCTOP)/scripts/recurse.mk

