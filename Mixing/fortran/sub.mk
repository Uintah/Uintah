#
# Makefile fragment for this subdirectory
# $Id$
#

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/Mixing/fortran

SRCS     += $(SRCDIR)/dqagpe.F $(SRCDIR)/dqelg.F $(SRCDIR)/dqk21.F \
	$(SRCDIR)/dqpsrt.F $(SRCDIR)/d1mach.F $(SRCDIR)/gammaln.F \
	$(SRCDIR)/eqlib.F $(SRCDIR)/stanlib.F $(SRCDIR)/cklib.F \
	$(SRCDIR)/ChemkinHacks.F

PSELIBS :=
#LIBS := -lftn -lm -lblas
#LIBS := -lftn -lm 

#FFLAGS += -g -O3 -OPT:IEEE_arithmetic=3 -CG:if_conversion=false:reverse_if_conversion=false -LNO:pf2=0 -avoid_gp_overflow -I$(SRCDIR)
#FFLAGS += -g 
FFLAGS +=

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2001/04/24 19:46:10  mcole
# prelim checking for package only compilation, scripts refferred to by SCIRUN_SCRIPTS variable
#
# Revision 1.1  2001/01/31 16:35:32  rawat
# Implemented mixing and reaction models for fire.
#
#
