# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Radiation/fortran

SRCS += \
	$(SRCDIR)/eco.F \
	$(SRCDIR)/eco2.F \
	$(SRCDIR)/eh2o.F \
	$(SRCDIR)/efuel.F \
	$(SRCDIR)/eico2.F \
	$(SRCDIR)/eih2o.F \
	$(SRCDIR)/find.F \
	$(SRCDIR)/fixradval.F \
	$(SRCDIR)/radarray.F \
	$(SRCDIR)/radcoef.F \
	$(SRCDIR)/radcal.F \
	$(SRCDIR)/rdombc.F \
	$(SRCDIR)/rdombmcalc.F \
	$(SRCDIR)/rdomflux.F \
	$(SRCDIR)/rdomsolve.F \
	$(SRCDIR)/rdomsrc.F \
	$(SRCDIR)/rdomvolq.F \
	$(SRCDIR)/rordr.F \
	$(SRCDIR)/rordrss.F \
	$(SRCDIR)/rordrtn.F \
	$(SRCDIR)/soot.F

PSELIBS := \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/radarray.o: $(SRCDIR)/radarray_fort.h
$(SRCDIR)/radcal.o: $(SRCDIR)/radcal_fort.h
$(SRCDIR)/radcoef.o: $(SRCDIR)/radcoef_fort.h
$(SRCDIR)/rordr.o: $(SRCDIR)/rordr_fort.h
$(SRCDIR)/rordrss.o: $(SRCDIR)/rordrss_fort.h
$(SRCDIR)/rordrtn.o: $(SRCDIR)/rordrtn_fort.h
$(SRCDIR)/rdombc.o: $(SRCDIR)/rdombc_fort.h
$(SRCDIR)/rdombmcalc.o: $(SRCDIR)/rdombmcalc_fort.h
$(SRCDIR)/rdomsolve.o: $(SRCDIR)/rdomsolve_fort.h
$(SRCDIR)/rdomsrc.o: $(SRCDIR)/rdomsrc_fort.h
$(SRCDIR)/rdomflux.o: $(SRCDIR)/rdomflux_fort.h
$(SRCDIR)/rdomvolq.o: $(SRCDIR)/rdomvolq_fort.h



