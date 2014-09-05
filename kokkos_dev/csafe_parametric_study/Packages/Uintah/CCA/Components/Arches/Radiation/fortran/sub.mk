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
	$(SRCDIR)/radwsgg.F \
	$(SRCDIR)/rdombc.F \
	$(SRCDIR)/rdombmcalc.F \
	$(SRCDIR)/rdomflux.F \
	$(SRCDIR)/rdomsolve.F \
	$(SRCDIR)/rdomsrc.F \
	$(SRCDIR)/rdomvolq.F \
	$(SRCDIR)/rordr.F \
	$(SRCDIR)/rordrss.F \
	$(SRCDIR)/rordrtn.F \
	$(SRCDIR)/rshsolve.F \
	$(SRCDIR)/rshresults.F \
	$(SRCDIR)/soot.F

PSELIBS := 

LIBS := $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/radarray.$(OBJEXT): $(SRCDIR)/radarray_fort.h
$(SRCDIR)/radcal.$(OBJEXT): $(SRCDIR)/radcal_fort.h
$(SRCDIR)/radcoef.$(OBJEXT): $(SRCDIR)/radcoef_fort.h
$(SRCDIR)/radwsgg.$(OBJEXT): $(SRCDIR)/radwsgg_fort.h
$(SRCDIR)/rordr.$(OBJEXT): $(SRCDIR)/rordr_fort.h
$(SRCDIR)/rordrss.$(OBJEXT): $(SRCDIR)/rordrss_fort.h
$(SRCDIR)/rordrtn.$(OBJEXT): $(SRCDIR)/rordrtn_fort.h
$(SRCDIR)/rdombc.$(OBJEXT): $(SRCDIR)/rdombc_fort.h
$(SRCDIR)/rdombmcalc.$(OBJEXT): $(SRCDIR)/rdombmcalc_fort.h
$(SRCDIR)/rdomsolve.$(OBJEXT): $(SRCDIR)/rdomsolve_fort.h
$(SRCDIR)/rdomsrc.$(OBJEXT): $(SRCDIR)/rdomsrc_fort.h
$(SRCDIR)/rdomflux.$(OBJEXT): $(SRCDIR)/rdomflux_fort.h
$(SRCDIR)/rdomvolq.$(OBJEXT): $(SRCDIR)/rdomvolq_fort.h
$(SRCDIR)/rshsolve.$(OBJEXT): $(SRCDIR)/rshsolve_fort.h
$(SRCDIR)/rshresults.$(OBJEXT): $(SRCDIR)/rshresults_fort.h

