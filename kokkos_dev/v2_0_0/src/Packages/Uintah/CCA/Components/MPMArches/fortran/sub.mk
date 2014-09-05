# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches/fortran

SRCS += $(SRCDIR)/collect_drag_cc.F \
	$(SRCDIR)/collect_scalar_fctocc.F \
	$(SRCDIR)/energy_exchange_term.F \
	$(SRCDIR)/interp_centertoface.F \
	$(SRCDIR)/momentum_exchange_term_continuous_cc.F \
	$(SRCDIR)/pressure_force.F \
	$(SRCDIR)/read_complex_geometry.F \
	$(SRCDIR)/taucal_cc.F \
	$(SRCDIR)/walmom_cc.F

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/collect_drag_cc.o: $(SRCDIR)/collect_drag_cc_fort.h
$(SRCDIR)/collect_scalar_fctocc.o: $(SRCDIR)/collect_scalar_fctocc_fort.h
$(SRCDIR)/energy_exchange_term.o: $(SRCDIR)/energy_exchange_term_fort.h
$(SRCDIR)/interp_centertoface.o: $(SRCDIR)/interp_centertoface_fort.h
$(SRCDIR)/momentum_exchange_term_continuous_cc.o: $(SRCDIR)/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/pressure_force.o: $(SRCDIR)/pressure_force_fort.h
$(SRCDIR)/read_complex_geometry.o: $(SRCDIR)/read_complex_geometry_fort.h
$(SRCDIR)/taucal_cc.o: $(SRCDIR)/taucal_cc_fort.h
$(SRCDIR)/walmom_cc.o: $(SRCDIR)/walmom_cc_fort.h
