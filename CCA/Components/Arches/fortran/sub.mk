# Makefile fragment for this subdirectory

#include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/fortran

SRCS     += $(SRCDIR)/init.F $(SRCDIR)/initScal.F $(SRCDIR)/celltypeInit.F \
	$(SRCDIR)/cellg.F $(SRCDIR)/areain.F $(SRCDIR)/profv.F \
	$(SRCDIR)/profscalar.F  $(SRCDIR)/smagmodel.F $(SRCDIR)/scalarvarmodel.F \
	$(SRCDIR)/calpbc.F \
	$(SRCDIR)/inlbcs.F $(SRCDIR)/uvelcoef.F $(SRCDIR)/vvelcoef.F \
	$(SRCDIR)/wvelcoef.F $(SRCDIR)/uvelsrc.F $(SRCDIR)/vvelsrc.F \
	$(SRCDIR)/wvelsrc.F $(SRCDIR)/arrass.F $(SRCDIR)/mascal.F \
	$(SRCDIR)/mascal_scalar.F \
	$(SRCDIR)/apcal.F $(SRCDIR)/apcal_vel.F $(SRCDIR)/prescoef.F \
	$(SRCDIR)/pressrc.F \
	$(SRCDIR)/bcuvel.F $(SRCDIR)/bcvvel.F $(SRCDIR)/bcwvel.F  \
	$(SRCDIR)/bcpress.F $(SRCDIR)/symbcs.F \
	$(SRCDIR)/prdbc1.F $(SRCDIR)/prdbc2.F $(SRCDIR)/wallbc.F \
	$(SRCDIR)/fixval.F \
	$(SRCDIR)/scalcoef.F $(SRCDIR)/coeffb.F $(SRCDIR)/rmean.F \
	$(SRCDIR)/addpressgrad.F $(SRCDIR)/calcpressgrad.F \
	$(SRCDIR)/addpressuregrad.F $(SRCDIR)/addtranssrc.F \
	$(SRCDIR)/bcscalar.F \
	$(SRCDIR)/scalsrc.F \
	$(SRCDIR)/rescal.F \
	$(SRCDIR)/arrl1.F $(SRCDIR)/underelax.F $(SRCDIR)/linegs.F \
	$(SRCDIR)/normpress.F $(SRCDIR)/explicit.F \
	$(SRCDIR)/mmcelltypeinit.F \
	$(SRCDIR)/mmmomsrc.F $(SRCDIR)/mmbcvelocity.F $(SRCDIR)/mmwallbc.F \
	$(SRCDIR)/mm_modify_prescoef.F \
	$(SRCDIR)/add_hydrostatic_term_topressure.F

# SRCS     += $(SRCDIR)/apcal.F $(SRCDIR)/areain.F $(SRCDIR)/arradd.F \
#	$(SRCDIR)/arrass.F $(SRCDIR)/arrcop.F $(SRCDIR)/arrl1.F \
#	$(SRCDIR)/arrmax.F $(SRCDIR)/assign.F $(SRCDIR)/bcp.F \
#	$(SRCDIR)/bcpt.F $(SRCDIR)/bcted.F $(SRCDIR)/bctke.F \
#	$(SRCDIR)/bcup.F $(SRCDIR)/bcvp.F $(SRCDIR)/bcwp.F \
#	$(SRCDIR)/caleps.F $(SRCDIR)/calpbc.F $(SRCDIR)/calscf.F \
#	$(SRCDIR)/cellg.F $(SRCDIR)/clip.F $(SRCDIR)/epsave.F \
#	$(SRCDIR)/erchek.F $(SRCDIR)/eval.F \
#	$(SRCDIR)/fncd.F $(SRCDIR)/gaxpy.F $(SRCDIR)/gcopy.F \
#	$(SRCDIR)/gdot.F $(SRCDIR)/geomin.F $(SRCDIR)/ggemv.F \
#	$(SRCDIR)/gminit.F $(SRCDIR)/gnrm2.F $(SRCDIR)/grdgrf.F \
#	$(SRCDIR)/grid.F $(SRCDIR)/grot.F $(SRCDIR)/grotg.F \
#	$(SRCDIR)/gscal.F $(SRCDIR)/gtrsv.F $(SRCDIR)/init.F \
#	$(SRCDIR)/inketm.F $(SRCDIR)/inlbcs.F $(SRCDIR)/intgrt.F \
#	$(SRCDIR)/invar.F $(SRCDIR)/linegs.F $(SRCDIR)/lisolv.F \
#	$(SRCDIR)/loglaw.F $(SRCDIR)/matvec.F $(SRCDIR)/mixltm.F \
#	$(SRCDIR)/omgcal.F $(SRCDIR)/pdep.F $(SRCDIR)/pprops.F \
#	$(SRCDIR)/prcf.F $(SRCDIR)/prdbc1.F $(SRCDIR)/prdbc2.F \
#	$(SRCDIR)/prec.F  $(SRCDIR)/props.F \
#	$(SRCDIR)/reade.F $(SRCDIR)/rescal.F $(SRCDIR)/resid1.F \
#	$(SRCDIR)/rite0.F $(SRCDIR)/root.F $(SRCDIR)/scale_factors.F \
#	$(SRCDIR)/solve.F $(SRCDIR)/symbcs.F $(SRCDIR)/wallbc.F \
#	$(SRCDIR)/cputim_sun.F
PSELIBS :=
#LIBS := -lftn -lm -lblas
#LIBS := -lftn -lm 

#FFLAGS += -g -O3 -OPT:IEEE_arithmetic=3 -CG:if_conversion=false:reverse_if_conversion=false -LNO:pf2=0 -avoid_gp_overflow -I$(SRCDIR)
FFLAGS += -g 

#include $(SRCTOP)/scripts/smallso_epilogue.mk

