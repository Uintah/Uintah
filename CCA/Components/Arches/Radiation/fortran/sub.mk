# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Radiation/fortran

SRCS += \
	$(SRCDIR)/eico2.F \
	$(SRCDIR)/eih2o.F \
	$(SRCDIR)/find.F \
	$(SRCDIR)/radcoef.F \
	$(SRCDIR)/rdombc.F \
	$(SRCDIR)/rdomcintm.F \
	$(SRCDIR)/rdomflux.F \
	$(SRCDIR)/rdominit.F \
	$(SRCDIR)/rdomr.F \
	$(SRCDIR)/rdomsolve.F \
	$(SRCDIR)/rdomsrc.F \
	$(SRCDIR)/rordr.F \
	$(SRCDIR)/rqacc.F \
	$(SRCDIR)/rqiew.F \
	$(SRCDIR)/rqins.F \
	$(SRCDIR)/rqitb.F \
	$(SRCDIR)/rxibc.F \
	$(SRCDIR)/ryibc.F \
	$(SRCDIR)/rzibc.F

PSELIBS := \
	Packages/Uintah/Core/Parallel    \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/CCA/Ports        \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Core/Thread 			 \
	Core/Geometry                    \
	Core/Exceptions

LIBS := $(FLIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


$(SRCDIR)/radcoef.o: $(SRCDIR)/radcoef_fort.h
$(SRCDIR)/rdomr.o: $(SRCDIR)/rdomr_fort.h
$(SRCDIR)/rordr.o: $(SRCDIR)/rordr_fort.h
$(SRCDIR)/rdombc.o: $(SRCDIR)/rdombc_fort.h
$(SRCDIR)/rdomsolve.o: $(SRCDIR)/rdomsolve_fort.h
$(SRCDIR)/rdomsrc.o: $(SRCDIR)/rdomsrc_fort.h
$(SRCDIR)/rdomflux.o: $(SRCDIR)/rdomflux_fort.h

