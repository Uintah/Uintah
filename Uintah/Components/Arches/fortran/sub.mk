#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/Arches/fortran

SRCS     += $(SRCDIR)/apcal.f $(SRCDIR)/areain.f $(SRCDIR)/arradd.f \
	$(SRCDIR)/arrass.f $(SRCDIR)/arrcop.f $(SRCDIR)/arrl1.f \
	$(SRCDIR)/arrmax.f $(SRCDIR)/assign.f $(SRCDIR)/bcp.f \
	$(SRCDIR)/bcpt.f $(SRCDIR)/bcted.f $(SRCDIR)/bctke.f \
	$(SRCDIR)/bcup.f $(SRCDIR)/bcvp.f $(SRCDIR)/bcwp.f \
	$(SRCDIR)/caleps.f $(SRCDIR)/calpbc.f $(SRCDIR)/calscf.f \
	$(SRCDIR)/cellg.f $(SRCDIR)/clip.f $(SRCDIR)/epsave.f \
	$(SRCDIR)/erchek.f $(SRCDIR)/eval.f $(SRCDIR)/fixval.f \
	$(SRCDIR)/fncd.f $(SRCDIR)/gaxpy.f $(SRCDIR)/gcopy.f \
	$(SRCDIR)/gdot.f $(SRCDIR)/geomin.f $(SRCDIR)/ggemv.f \
	$(SRCDIR)/gminit.f $(SRCDIR)/gnrm2.f $(SRCDIR)/grdgrf.f \
	$(SRCDIR)/grid.f $(SRCDIR)/grot.f $(SRCDIR)/grotg.f \
	$(SRCDIR)/gscal.f $(SRCDIR)/gtrsv.f $(SRCDIR)/init.f \
	$(SRCDIR)/inketm.f $(SRCDIR)/inlbcs.f $(SRCDIR)/intgrt.f \
	$(SRCDIR)/invar.f $(SRCDIR)/linegs.f $(SRCDIR)/lisolv.f \
	$(SRCDIR)/loglaw.f $(SRCDIR)/matvec.f $(SRCDIR)/mixltm.f \
	$(SRCDIR)/omgcal.f $(SRCDIR)/pdep.f $(SRCDIR)/pprops.f \
	$(SRCDIR)/prcf.f $(SRCDIR)/prdbc1.f $(SRCDIR)/prdbc2.f \
	$(SRCDIR)/prec.f $(SRCDIR)/profv.f $(SRCDIR)/props.f \
	$(SRCDIR)/reade.f $(SRCDIR)/rescal.f $(SRCDIR)/resid1.f \
	$(SRCDIR)/rite0.f $(SRCDIR)/root.f $(SRCDIR)/scale_factors.f \
	$(SRCDIR)/solve.f $(SRCDIR)/symbcs.f $(SRCDIR)/wallbc.f
PSELIBS :=
LIBS := -lftn -lm -lblas

FFLAGS := $(FFLAGS) -I$(SRCDIR)

$(SRCDIR)/apcal.f : $(SRCDIR)/apcal.F
	cpp $(SRCDIR)/apcal.F -o $(SRCDIR)/apcal.f

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/05/11 20:10:11  dav
# adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
#
# Revision 1.1  2000/04/13 20:06:30  sparker
# Makefile fragment for the subdir
#
#
