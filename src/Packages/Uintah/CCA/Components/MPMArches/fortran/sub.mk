# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/MPMArches/fortran

SRCS     += $(SRCDIR)/arrass.F \
	$(SRCDIR)/collect_drag_cc.F \
	$(SRCDIR)/interp_centertoface.F \
	$(SRCDIR)/interp_facetocenter.F \
	$(SRCDIR)/momentum_exchange_term_continuous_cc.F \
	$(SRCDIR)/pressure_force.F \
	$(SRCDIR)/redistribute_dragforce_cc.F \
	$(SRCDIR)/taucal_cc.F \
	$(SRCDIR)/walmom_cc.F

PSELIBS :=
#LIBS := -lftn -lm -lblas
#LIBS := -lftn -lm 

#FFLAGS += -g -O3 -OPT:IEEE_arithmetic=3 -CG:if_conversion=false:reverse_if_conversion=false -LNO:pf2=0 -avoid_gp_overflow -I$(SRCDIR)
FFLAGS += -g 

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

