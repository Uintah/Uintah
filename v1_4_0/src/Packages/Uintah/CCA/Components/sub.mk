# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/CCA/Components

SUBDIRS := \
	$(SRCDIR)/DataArchiver \
	$(SRCDIR)/Examples \
	$(SRCDIR)/Schedulers \
	$(SRCDIR)/SimulationController \
	$(SRCDIR)/MPM \
	$(SRCDIR)/ICE \
	$(SRCDIR)/MPMICE \
	$(SRCDIR)/MPMArches \
	$(SRCDIR)/Arches \
	$(SRCDIR)/ProblemSpecification \
	$(SRCDIR)/HETransformation

include $(SCIRUN_SCRIPTS)/recurse.mk

