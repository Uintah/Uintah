#
# Makefile fragment for this subdirectory
# $Id$
#

#include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/Arches/fortran

SRCS     += $(SRCDIR)/apcal.F $(SRCDIR)/areain.F $(SRCDIR)/arradd.F \
	$(SRCDIR)/arrass.F $(SRCDIR)/arrcop.F $(SRCDIR)/arrl1.F \
	$(SRCDIR)/arrmax.F $(SRCDIR)/assign.F $(SRCDIR)/bcp.F \
	$(SRCDIR)/bcpt.F $(SRCDIR)/bcted.F $(SRCDIR)/bctke.F \
	$(SRCDIR)/bcup.F $(SRCDIR)/bcvp.F $(SRCDIR)/bcwp.F \
	$(SRCDIR)/caleps.F $(SRCDIR)/calpbc.F $(SRCDIR)/calscf.F \
	$(SRCDIR)/cellg.F $(SRCDIR)/clip.F $(SRCDIR)/epsave.F \
	$(SRCDIR)/erchek.F $(SRCDIR)/eval.F $(SRCDIR)/fixval.F \
	$(SRCDIR)/fncd.F $(SRCDIR)/gaxpy.F $(SRCDIR)/gcopy.F \
	$(SRCDIR)/gdot.F $(SRCDIR)/geomin.F $(SRCDIR)/ggemv.F \
	$(SRCDIR)/gminit.F $(SRCDIR)/gnrm2.F $(SRCDIR)/grdgrf.F \
	$(SRCDIR)/grid.F $(SRCDIR)/grot.F $(SRCDIR)/grotg.F \
	$(SRCDIR)/gscal.F $(SRCDIR)/gtrsv.F $(SRCDIR)/init.F \
	$(SRCDIR)/inketm.F $(SRCDIR)/inlbcs.F $(SRCDIR)/intgrt.F \
	$(SRCDIR)/invar.F $(SRCDIR)/linegs.F $(SRCDIR)/lisolv.F \
	$(SRCDIR)/loglaw.F $(SRCDIR)/matvec.F $(SRCDIR)/mixltm.F \
	$(SRCDIR)/omgcal.F $(SRCDIR)/pdep.F $(SRCDIR)/pprops.F \
	$(SRCDIR)/prcf.F $(SRCDIR)/prdbc1.F $(SRCDIR)/prdbc2.F \
	$(SRCDIR)/prec.F $(SRCDIR)/profv.F $(SRCDIR)/props.F \
	$(SRCDIR)/reade.F $(SRCDIR)/rescal.F $(SRCDIR)/resid1.F \
	$(SRCDIR)/rite0.F $(SRCDIR)/root.F $(SRCDIR)/scale_factors.F \
	$(SRCDIR)/solve.F $(SRCDIR)/symbcs.F $(SRCDIR)/wallbc.F \
	$(SRCDIR)/cputim_sun.F
PSELIBS :=
#LIBS := -lftn -lm -lblas

FFLAGS := -O3 -OPT:IEEE_arithmetic=3 -CG:if_conversion=false:reverse_i
f_conversion=false -LNO:pf2=0 -avoid_gp_overflow -mp -I$(SRCDIR)

#include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/05/17 21:51:18  bbanerje
# Added the file containing _cputim.
#
# Revision 1.3  2000/05/17 21:36:44  bbanerje
# Changed .f to .F in SRCS and added FFLAGS specfic to these .f files.
#
# Revision 1.2  2000/05/11 20:10:11  dav
# adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
#
# Revision 1.1  2000/04/13 20:06:30  sparker
# Makefile fragment for the subdir
#
#
