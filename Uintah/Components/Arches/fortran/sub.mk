#
# Makefile fragment for this subdirectory
# $Id$
#

#include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/Arches/fortran

SRCS     += $(SRCDIR)/init.F $(SRCDIR)/initScal.F $(SRCDIR)/celltypeInit.F \
	$(SRCDIR)/cellg.F $(SRCDIR)/areain.F $(SRCDIR)/profv.F \
	$(SRCDIR)/profscalar.F  $(SRCDIR)/smagmodel.F $(SRCDIR)/calpbc.F \
	$(SRCDIR)/inlbcs.F $(SRCDIR)/uvelcoef.F $(SRCDIR)/vvelcoef.F \
	$(SRCDIR)/wvelcoef.F $(SRCDIR)/uvelsrc.F $(SRCDIR)/vvelsrc.F \
	$(SRCDIR)/wvelsrc.F $(SRCDIR)/arrass.F $(SRCDIR)/mascal.F \
	$(SRCDIR)/apcal.F $(SRCDIR)/prescoef.F $(SRCDIR)/pressrc.F \
	$(SRCDIR)/bcuvel.F $(SRCDIR)/bcpress.F $(SRCDIR)/symbcs.F \
	$(SRCDIR)/prdbc1.F $(SRCDIR)/prdbc2.F $(SRCDIR)/wallbc.F
# SRCS     += $(SRCDIR)/apcal.F $(SRCDIR)/areain.F $(SRCDIR)/arradd.F \
#	$(SRCDIR)/arrass.F $(SRCDIR)/arrcop.F $(SRCDIR)/arrl1.F \
#	$(SRCDIR)/arrmax.F $(SRCDIR)/assign.F $(SRCDIR)/bcp.F \
#	$(SRCDIR)/bcpt.F $(SRCDIR)/bcted.F $(SRCDIR)/bctke.F \
#	$(SRCDIR)/bcup.F $(SRCDIR)/bcvp.F $(SRCDIR)/bcwp.F \
#	$(SRCDIR)/caleps.F $(SRCDIR)/calpbc.F $(SRCDIR)/calscf.F \
#	$(SRCDIR)/cellg.F $(SRCDIR)/clip.F $(SRCDIR)/epsave.F \
#	$(SRCDIR)/erchek.F $(SRCDIR)/eval.F $(SRCDIR)/fixval.F \
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

#
# $Log$
# Revision 1.25  2000/07/13 04:51:35  bbanerje
# Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
#
# Revision 1.24  2000/07/12 23:59:23  rawat
# added wall bc for u-velocity
#
# Revision 1.23  2000/07/12 23:23:25  bbanerje
# Added pressure source .. modified Kumar's version a bit.
#
# Revision 1.22  2000/07/12 22:15:05  bbanerje
# Added pressure Coef .. will do until Kumar's code is up and running
#
# Revision 1.21  2000/07/12 19:55:45  bbanerje
# Added apcal stuff in calcVelDiagonal
#
# Revision 1.20  2000/07/12 07:35:49  bbanerje
# Added stuff for mascal : Rawat: Labels and dataWarehouse in velsrc need to be corrected.
#
# Revision 1.19  2000/07/12 05:14:27  bbanerje
# Added vvelsrc and wvelsrc .. some changes to uvelsrc.
# Rawat :: Labels are getting hopelessly muddled unless we can do something
# about the time stepping thing.
#
# Revision 1.18  2000/07/11 15:46:30  rawat
# added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
#
# Revision 1.17  2000/07/08 23:08:56  bbanerje
# Added vvelcoef and wvelcoef ..
# Rawat check the ** WARNING ** tags in these files for possible problems.
#
# Revision 1.16  2000/07/08 08:03:37  bbanerje
# Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
# made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
# to Arches::AE .. doesn't like enums in templates apparently.
#
# Revision 1.15  2000/07/07 23:07:48  rawat
# added inlet bc's
#
# Revision 1.14  2000/07/03 05:30:22  bbanerje
# Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
#
# Revision 1.13  2000/07/01 05:21:01  bbanerje
# Changed CellInformation calcs for Turbulence model requirements ..
# CellInformation still needs work.
#
# Revision 1.12  2000/06/30 04:19:18  rawat
# added turbulence model and compute properties
#
# Revision 1.11  2000/06/20 20:42:38  rawat
# added some more boundary stuff and modified interface to IntVector. Before
# compiling the code you need to update /SCICore/Geometry/IntVector.h
#
# Revision 1.10  2000/06/15 22:13:24  rawat
# modified boundary stuff
#
# Revision 1.9  2000/06/14 23:07:59  bbanerje
# Added celltypeInit.F and sub.mk
#
# Revision 1.8  2000/06/14 21:25:25  jas
# removed celltypeInit.F from compilation.
#
# Revision 1.7  2000/06/14 20:40:53  rawat
# modified boundarycondition for physical boundaries and
# added CellInformation class
#
# Revision 1.6  2000/06/13 20:51:45  bbanerje
# Fortran flags not overwritten now (but still not done thru configure)
#
# Revision 1.5  2000/06/12 21:30:03  bbanerje
# Added first Fortran routines, added Stencil Matrix where needed,
# removed unnecessary CCVariables (e.g., sources etc.)
#
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
