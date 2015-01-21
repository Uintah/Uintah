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
	$(SRCDIR)/read_complex_geometry_walls.F \
	$(SRCDIR)/taucal_cc.F \
	$(SRCDIR)/walmom_cc.F

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/collect_drag_cc.$(OBJEXT): $(SRCDIR)/collect_drag_cc_fort.h
$(SRCDIR)/collect_scalar_fctocc.$(OBJEXT): $(SRCDIR)/collect_scalar_fctocc_fort.h
$(SRCDIR)/energy_exchange_term.$(OBJEXT): $(SRCDIR)/energy_exchange_term_fort.h
$(SRCDIR)/interp_centertoface.$(OBJEXT): $(SRCDIR)/interp_centertoface_fort.h
$(SRCDIR)/momentum_exchange_term_continuous_cc.$(OBJEXT): $(SRCDIR)/momentum_exchange_term_continuous_cc_fort.h
$(SRCDIR)/pressure_force.$(OBJEXT): $(SRCDIR)/pressure_force_fort.h
$(SRCDIR)/read_complex_geometry.$(OBJEXT): $(SRCDIR)/read_complex_geometry_fort.h
$(SRCDIR)/read_complex_geometry_walls.$(OBJEXT): $(SRCDIR)/read_complex_geometry_walls_fort.h
$(SRCDIR)/taucal_cc.$(OBJEXT): $(SRCDIR)/taucal_cc_fort.h
$(SRCDIR)/walmom_cc.$(OBJEXT): $(SRCDIR)/walmom_cc_fort.h
