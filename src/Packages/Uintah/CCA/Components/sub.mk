# Makefile fragment for this subdirectory

SRCDIR := Packages/Uintah/CCA/Components

SUBDIRS := \
	$(SRCDIR)/DataArchiver \
	$(SRCDIR)/Examples \
	$(SRCDIR)/Models \
	$(SRCDIR)/Schedulers \
	$(SRCDIR)/SimulationController \
	$(SRCDIR)/MPM \
	$(SRCDIR)/ICE \
	$(SRCDIR)/MPMICE \
	$(SRCDIR)/MPMArches \
	$(SRCDIR)/Arches \
	$(SRCDIR)/Arches/fortran \
	$(SRCDIR)/Arches/Mixing \
	$(SRCDIR)/Arches/Radiation \
	$(SRCDIR)/Arches/Radiation/fortran \
	$(SRCDIR)/ProblemSpecification \
	$(SRCDIR)/HETransformation \
	$(SRCDIR)/PatchCombiner \
	$(SRCDIR)/Solvers

include $(SCIRUN_SCRIPTS)/recurse.mk
