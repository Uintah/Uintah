# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches/fortran

SRCS += $(SRCDIR)/collect_drag_cc.F \
	$(SRCDIR)/interp_centertoface.F \
	$(SRCDIR)/momentum_exchange_term_continuous_cc.F \
	$(SRCDIR)/pressure_force.F \

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/interp_centertoface.o: $(SRCDIR)/interp_centertoface_fort.h
$(SRCDIR)/collect_drag_cc.o: $(SRCDIR)/collect_drag_cc_fort.h
$(SRCDIR)/pressure_force.o: $(SRCDIR)/pressure_force_fort.h
$(SRCDIR)/momentum_exchange_term_continuous_cc.o: $(SRCDIR)/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/redistribute_dragforce_cc.o: $(SRCDIR)/redistribute_dragforce_cc_fort.h

