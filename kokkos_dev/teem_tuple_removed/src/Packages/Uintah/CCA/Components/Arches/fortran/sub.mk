# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/fortran

SRCS += \
	$(SRCDIR)/add_hydrostatic_term_topressure.F \
	$(SRCDIR)/add_mm_enth_src.F \
	$(SRCDIR)/addpressgrad.F \
	$(SRCDIR)/addpressuregrad.F \
	$(SRCDIR)/addtranssrc.F \
	$(SRCDIR)/apcal.F \
	$(SRCDIR)/apcal_vel.F \
	$(SRCDIR)/areain.F \
	$(SRCDIR)/arrass.F \
	$(SRCDIR)/arrl1.F \
	$(SRCDIR)/bcenthalpy.F \
	$(SRCDIR)/bcinout.F \
	$(SRCDIR)/inlpresbcinout.F \
	$(SRCDIR)/bcpress.F \
	$(SRCDIR)/bcscalar.F \
	$(SRCDIR)/bcuvel.F \
	$(SRCDIR)/bcvvel.F \
	$(SRCDIR)/bcwvel.F \
	$(SRCDIR)/calcpressgrad.F \
	$(SRCDIR)/calpbc.F \
        $(SRCDIR)/hatvelcalpbc.F \
	$(SRCDIR)/cellg.F \
	$(SRCDIR)/celltypeInit.F \
	$(SRCDIR)/computeVel.F \
	$(SRCDIR)/denaccum.F \
	$(SRCDIR)/enthalpyradflux.F \
	$(SRCDIR)/enthalpyradsrc.F \
	$(SRCDIR)/enthalpyradthinsrc.F \
	$(SRCDIR)/enthalpyradwallbc.F \
	$(SRCDIR)/explicit.F \
	$(SRCDIR)/explicit_vel.F \
	$(SRCDIR)/explicit_velocity.F \
	$(SRCDIR)/fixval.F \
	$(SRCDIR)/fixval_trans.F \
	$(SRCDIR)/init.F \
	$(SRCDIR)/initScal.F \
	$(SRCDIR)/inlbcs.F \
	$(SRCDIR)/intrusion_computevel.F \
	$(SRCDIR)/linegs.F \
	$(SRCDIR)/mascal.F \
	$(SRCDIR)/mascal_scalar.F \
	$(SRCDIR)/uvelcoeffupdate.F \
	$(SRCDIR)/mm_computevel.F\
	$(SRCDIR)/mm_explicit.F\
	$(SRCDIR)/mm_explicit_oldvalue.F\
	$(SRCDIR)/mm_explicit_vel.F\
	$(SRCDIR)/mm_modify_prescoef.F \
	$(SRCDIR)/mmbcvelocity.F \
	$(SRCDIR)/mmbcvelocity_momex.F \
	$(SRCDIR)/mmbcenthalpy_energyex.F \
	$(SRCDIR)/mmcelltypeinit.F \
	$(SRCDIR)/mmmomsrc.F \
	$(SRCDIR)/mmwallbc.F \
	$(SRCDIR)/mmwallbc_trans.F \
	$(SRCDIR)/normpress.F \
	$(SRCDIR)/outarea.F \
	$(SRCDIR)/outletbc.F \
	$(SRCDIR)/outletbcenth.F \
	$(SRCDIR)/outletbcrscal.F \
	$(SRCDIR)/prescoef.F \
        $(SRCDIR)/prescoef_var.F \
        $(SRCDIR)/pressrcpred_var.F \
	$(SRCDIR)/pressrc.F \
	$(SRCDIR)/pressrccorr.F \
	$(SRCDIR)/pressrcpred.F \
	$(SRCDIR)/profscalar.F \
	$(SRCDIR)/profv.F \
	$(SRCDIR)/rescal.F \
	$(SRCDIR)/scalarvarmodel.F \
	$(SRCDIR)/scalcoef.F \
	$(SRCDIR)/scalsrc.F \
	$(SRCDIR)/smagmodel.F \
	$(SRCDIR)/symbcs.F \
	$(SRCDIR)/underelax.F \
	$(SRCDIR)/uvelcoef.F \
	$(SRCDIR)/uvelsrc.F \
	$(SRCDIR)/vvelcoef.F \
	$(SRCDIR)/vvelsrc.F \
	$(SRCDIR)/wallbc.F \
	$(SRCDIR)/wvelcoef.F \
	$(SRCDIR)/wvelsrc.F \
	$(SRCDIR)/dynamic_1loop.F \
	$(SRCDIR)/dynamic_2loop.F \
	$(SRCDIR)/dynamic_3loop.F

PSELIBS := \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/Core/Disclosure  \
	Packages/Uintah/Core/Exceptions  \
	Core/Exceptions

LIBS := $(XML_LIBRARY) $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


$(SRCDIR)/add_hydrostatic_term_topressure.o: $(SRCDIR)/add_hydrostatic_term_topressure_fort.h
$(SRCDIR)/add_mm_enth_src.o: $(SRCDIR)/add_mm_enth_src_fort.h
$(SRCDIR)/addpressgrad.o: $(SRCDIR)/addpressgrad_fort.h
$(SRCDIR)/addpressuregrad.o: $(SRCDIR)/addpressuregrad_fort.h
$(SRCDIR)/addtranssrc.o: $(SRCDIR)/addtranssrc_fort.h
$(SRCDIR)/apcal.o: $(SRCDIR)/apcal_fort.h
$(SRCDIR)/apcal_vel.o: $(SRCDIR)/apcal_vel_fort.h
$(SRCDIR)/areain.o: $(SRCDIR)/areain_fort.h
$(SRCDIR)/arrass.o: $(SRCDIR)/arrass_fort.h
$(SRCDIR)/arrl1.o: $(SRCDIR)/arrl1_fort.h
$(SRCDIR)/bcenthalpy.o: $(SRCDIR)/bcenthalpy_fort.h
$(SRCDIR)/bcinout.o: $(SRCDIR)/bcinout_fort.h
$(SRCDIR)/inlpresbcinout.o: $(SRCDIR)/inlpresbcinout_fort.h
$(SRCDIR)/bcpress.o: $(SRCDIR)/bcpress_fort.h
$(SRCDIR)/bcscalar.o: $(SRCDIR)/bcscalar_fort.h
$(SRCDIR)/bcuvel.o: $(SRCDIR)/bcuvel_fort.h
$(SRCDIR)/bcvvel.o: $(SRCDIR)/bcvvel_fort.h
$(SRCDIR)/bcwvel.o: $(SRCDIR)/bcwvel_fort.h
$(SRCDIR)/calcpressgrad.o: $(SRCDIR)/calcpressgrad_fort.h
$(SRCDIR)/calpbc.o: $(SRCDIR)/calpbc_fort.h
$(SRCDIR)/cellg.o: $(SRCDIR)/cellg_fort.h
$(SRCDIR)/celltypeInit.o: $(SRCDIR)/celltypeInit_fort.h
$(SRCDIR)/coeffb.o: $(SRCDIR)/coeffb_fort.h
$(SRCDIR)/computeVel.o: $(SRCDIR)/computeVel_fort.h
$(SRCDIR)/denaccum.o: $(SRCDIR)/denaccum_fort.h
$(SRCDIR)/enthalpyradflux.o: $(SRCDIR)/enthalpyradflux_fort.h
$(SRCDIR)/enthalpyradsrc.o: $(SRCDIR)/enthalpyradsrc_fort.h
$(SRCDIR)/enthalpyradthinsrc.o: $(SRCDIR)/enthalpyradthinsrc_fort.h
$(SRCDIR)/enthalpyradwallbc.o: $(SRCDIR)/enthalpyradwallbc_fort.h
$(SRCDIR)/explicit.o: $(SRCDIR)/explicit_fort.h
$(SRCDIR)/explicit_vel.o: $(SRCDIR)/explicit_vel_fort.h
$(SRCDIR)/explicit_velocity.o: $(SRCDIR)/explicit_velocity_fort.h
$(SRCDIR)/init.o: $(SRCDIR)/init_fort.h
$(SRCDIR)/initScal.o: $(SRCDIR)/initScal_fort.h
$(SRCDIR)/inlbcs.o: $(SRCDIR)/inlbcs_fort.h $(SRCDIR)/ramping.h
$(SRCDIR)/linegs.o: $(SRCDIR)/linegs_fort.h
$(SRCDIR)/mascal.o: $(SRCDIR)/mascal_fort.h
$(SRCDIR)/mascal_scalar.o: $(SRCDIR)/mascal_scalar_fort.h
$(SRCDIR)/mm_computevel.o: $(SRCDIR)/mm_computevel_fort.h
$(SRCDIR)/mm_explicit.o: $(SRCDIR)/mm_explicit_fort.h
$(SRCDIR)/mm_explicit_oldvalue.o: $(SRCDIR)/mm_explicit_oldvalue_fort.h
$(SRCDIR)/mm_explicit_vel.o: $(SRCDIR)/mm_explicit_vel_fort.h
$(SRCDIR)/mm_modify_prescoef.o: $(SRCDIR)/mm_modify_prescoef_fort.h
$(SRCDIR)/mmbcvelocity.o: $(SRCDIR)/mmbcvelocity_fort.h
$(SRCDIR)/mmcelltypeinit.o: $(SRCDIR)/mmcelltypeinit_fort.h
$(SRCDIR)/mmmomsrc.o: $(SRCDIR)/mmmomsrc_fort.h
$(SRCDIR)/mmwallbc.o: $(SRCDIR)/mmwallbc_fort.h
$(SRCDIR)/mmwallbc_trans.o: $(SRCDIR)/mmwallbc_trans_fort.h
$(SRCDIR)/normpress.o: $(SRCDIR)/normpress_fort.h
$(SRCDIR)/outarea.o: $(SRCDIR)/outarea_fort.h
$(SRCDIR)/outletbc.o: $(SRCDIR)/outletbc_fort.h
$(SRCDIR)/outletbcenth.o: $(SRCDIR)/outletbcenth_fort.h
$(SRCDIR)/outletbcrscal.o: $(SRCDIR)/outletbcrscal_fort.h
$(SRCDIR)/prdbc1.o: $(SRCDIR)/prdbc1_fort.h
$(SRCDIR)/prdbc2.o: $(SRCDIR)/prdbc2_fort.h
$(SRCDIR)/prescoef.o: $(SRCDIR)/prescoef_fort.h
$(SRCDIR)/prescoef_var.o: $(SRCDIR)/prescoef_var_fort.h
$(SRCDIR)/pressrc.o: $(SRCDIR)/pressrc_fort.h
$(SRCDIR)/pressrccorr.o: $(SRCDIR)/pressrccorr_fort.h
$(SRCDIR)/pressrcpred.o: $(SRCDIR)/pressrcpred_fort.h
$(SRCDIR)/pressrcpred_var.o: $(SRCDIR)/pressrcpred_var_fort.h
$(SRCDIR)/profscalar.o: $(SRCDIR)/profscalar_fort.h
$(SRCDIR)/profv.o: $(SRCDIR)/profv_fort.h $(SRCDIR)/ramping.h
$(SRCDIR)/rescal.o: $(SRCDIR)/rescal_fort.h
$(SRCDIR)/rmean.o: $(SRCDIR)/rmean_fort.h
$(SRCDIR)/scalarvarmodel.o: $(SRCDIR)/scalarvarmodel_fort.h
$(SRCDIR)/scalcoef.o: $(SRCDIR)/scalcoef_fort.h
$(SRCDIR)/scalsrc.o: $(SRCDIR)/scalsrc_fort.h
$(SRCDIR)/smagmodel.o: $(SRCDIR)/smagmodel_fort.h
$(SRCDIR)/symbcs.o: $(SRCDIR)/symbcs_fort.h
$(SRCDIR)/underelax.o: $(SRCDIR)/underelax_fort.h
$(SRCDIR)/uvelcoef.o: $(SRCDIR)/uvelcoef_fort.h
$(SRCDIR)/uvelsrc.o: $(SRCDIR)/uvelsrc_fort.h
$(SRCDIR)/vvelcoef.o: $(SRCDIR)/vvelcoef_fort.h
$(SRCDIR)/vvelsrc.o: $(SRCDIR)/vvelsrc_fort.h
$(SRCDIR)/wallbc.o: $(SRCDIR)/wallbc_fort.h
$(SRCDIR)/wvelcoef.o: $(SRCDIR)/wvelcoef_fort.h
$(SRCDIR)/wvelsrc.o: $(SRCDIR)/wvelsrc_fort.h
$(SRCDIR)/hatvelcalpbc.o: $(SRCDIR)/hatvelcalpbc_fort.h
$(SRCDIR)/uvelcoeffupdate.o: $(SRCDIR)/uvelcoeffupdate_fort.h
$(SRCDIR)/dynamic_1loop.o: $(SRCDIR)/dynamic_1loop_fort.h
$(SRCDIR)/dynamic_2loop.o: $(SRCDIR)/dynamic_2loop_fort.h
$(SRCDIR)/dynamic_3loop.o: $(SRCDIR)/dynamic_3loop_fort.h
$(SRCDIR)/intrusion_computevel.o: $(SRCDIR)/intrusion_computevel_fort.h
$(SRCDIR)/mmbcvelocity_momex.o: $(SRCDIR)/mmbcvelocity_momex_fort.h
$(SRCDIR)/mmbcenthalpy_energyex.o: $(SRCDIR)/mmbcenthalpy_energyex_fort.h

