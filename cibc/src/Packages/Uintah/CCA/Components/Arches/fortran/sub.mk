# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/CCA/Components/Arches/fortran

SRCS += \
	$(SRCDIR)/add_hydrostatic_term_topressure.F \
	$(SRCDIR)/add_mm_enth_src.F \
	$(SRCDIR)/apcal.F \
	$(SRCDIR)/apcal_vel.F \
	$(SRCDIR)/areain.F \
	$(SRCDIR)/arrass.F \
	$(SRCDIR)/bcenthalpy.F \
	$(SRCDIR)/inlpresbcinout.F \
	$(SRCDIR)/bcpress.F \
	$(SRCDIR)/bcscalar.F \
	$(SRCDIR)/bcuvel.F \
	$(SRCDIR)/bcvvel.F \
	$(SRCDIR)/bcwvel.F \
	$(SRCDIR)/cellg.F \
	$(SRCDIR)/celltypeInit.F \
	$(SRCDIR)/computeVel.F \
	$(SRCDIR)/enthalpyradthinsrc.F \
	$(SRCDIR)/explicit_scalar.F \
	$(SRCDIR)/explicit_vel.F \
	$(SRCDIR)/fixval.F \
	$(SRCDIR)/fixval_trans.F \
	$(SRCDIR)/inlbcs.F \
	$(SRCDIR)/intrusion_computevel.F \
	$(SRCDIR)/mascal.F \
	$(SRCDIR)/mascal_scalar.F \
	$(SRCDIR)/mm_computevel.F\
	$(SRCDIR)/mm_explicit.F\
	$(SRCDIR)/mm_explicit_oldvalue.F\
	$(SRCDIR)/mm_explicit_vel.F\
	$(SRCDIR)/mm_modify_prescoef.F \
	$(SRCDIR)/mmbcvelocity.F \
	$(SRCDIR)/mmbcvelocity_momex.F \
	$(SRCDIR)/mmbcenthalpy_energyex.F \
	$(SRCDIR)/mmenthalpywallbc.F \
	$(SRCDIR)/mmcelltypeinit.F \
	$(SRCDIR)/mmmomsrc.F \
	$(SRCDIR)/mmscalarwallbc.F \
	$(SRCDIR)/mmwallbc.F \
	$(SRCDIR)/mmwallbc_trans.F \
	$(SRCDIR)/normpress.F \
	$(SRCDIR)/prescoef.F \
        $(SRCDIR)/prescoef_var.F \
	$(SRCDIR)/pressrcpred.F \
        $(SRCDIR)/pressrcpred_var.F \
	$(SRCDIR)/profscalar.F \
	$(SRCDIR)/profv.F \
	$(SRCDIR)/scalarvarmodel.F \
	$(SRCDIR)/scalcoef.F \
	$(SRCDIR)/scalsrc.F \
	$(SRCDIR)/smagmodel.F \
	$(SRCDIR)/uvelcoef.F \
	$(SRCDIR)/uvelsrc.F \
	$(SRCDIR)/vvelcoef.F \
	$(SRCDIR)/vvelsrc.F \
	$(SRCDIR)/wallbc.F \
	$(SRCDIR)/wvelcoef.F \
	$(SRCDIR)/wvelsrc.F \
	$(SRCDIR)/inc_dynamic_1loop.F \
	$(SRCDIR)/inc_dynamic_2loop.F \
	$(SRCDIR)/inc_dynamic_3loop.F \
	$(SRCDIR)/comp_dynamic_1loop.F \
	$(SRCDIR)/comp_dynamic_2loop.F \
	$(SRCDIR)/comp_dynamic_3loop.F \
	$(SRCDIR)/comp_dynamic_4loop.F \
	$(SRCDIR)/comp_dynamic_5loop.F \
	$(SRCDIR)/comp_dynamic_6loop.F \
	$(SRCDIR)/comp_dynamic_7loop.F \
	$(SRCDIR)/comp_dynamic_8loop.F

PSELIBS := 

LIBS := $(F_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


$(SRCDIR)/add_hydrostatic_term_topressure.$(OBJEXT): $(SRCDIR)/add_hydrostatic_term_topressure_fort.h
$(SRCDIR)/add_mm_enth_src.$(OBJEXT): $(SRCDIR)/add_mm_enth_src_fort.h
$(SRCDIR)/apcal.$(OBJEXT): $(SRCDIR)/apcal_fort.h
$(SRCDIR)/apcal_vel.$(OBJEXT): $(SRCDIR)/apcal_vel_fort.h
$(SRCDIR)/areain.$(OBJEXT): $(SRCDIR)/areain_fort.h
$(SRCDIR)/arrass.$(OBJEXT): $(SRCDIR)/arrass_fort.h
$(SRCDIR)/bcenthalpy.$(OBJEXT): $(SRCDIR)/bcenthalpy_fort.h
$(SRCDIR)/inlpresbcinout.$(OBJEXT): $(SRCDIR)/inlpresbcinout_fort.h
$(SRCDIR)/bcpress.$(OBJEXT): $(SRCDIR)/bcpress_fort.h
$(SRCDIR)/bcscalar.$(OBJEXT): $(SRCDIR)/bcscalar_fort.h
$(SRCDIR)/bcuvel.$(OBJEXT): $(SRCDIR)/bcuvel_fort.h
$(SRCDIR)/bcvvel.$(OBJEXT): $(SRCDIR)/bcvvel_fort.h
$(SRCDIR)/bcwvel.$(OBJEXT): $(SRCDIR)/bcwvel_fort.h
$(SRCDIR)/cellg.$(OBJEXT): $(SRCDIR)/cellg_fort.h
$(SRCDIR)/celltypeInit.$(OBJEXT): $(SRCDIR)/celltypeInit_fort.h
$(SRCDIR)/computeVel.$(OBJEXT): $(SRCDIR)/computeVel_fort.h
$(SRCDIR)/enthalpyradthinsrc.$(OBJEXT): $(SRCDIR)/enthalpyradthinsrc_fort.h
$(SRCDIR)/explicit_scalar.$(OBJEXT): $(SRCDIR)/explicit_scalar_fort.h
$(SRCDIR)/explicit_vel.$(OBJEXT): $(SRCDIR)/explicit_vel_fort.h
$(SRCDIR)/inlbcs.$(OBJEXT): $(SRCDIR)/inlbcs_fort.h $(SRCDIR)/ramping.h
$(SRCDIR)/mascal.$(OBJEXT): $(SRCDIR)/mascal_fort.h
$(SRCDIR)/mascal_scalar.$(OBJEXT): $(SRCDIR)/mascal_scalar_fort.h
$(SRCDIR)/mm_computevel.$(OBJEXT): $(SRCDIR)/mm_computevel_fort.h
$(SRCDIR)/mm_explicit.$(OBJEXT): $(SRCDIR)/mm_explicit_fort.h
$(SRCDIR)/mm_explicit_oldvalue.$(OBJEXT): $(SRCDIR)/mm_explicit_oldvalue_fort.h
$(SRCDIR)/mm_explicit_vel.$(OBJEXT): $(SRCDIR)/mm_explicit_vel_fort.h
$(SRCDIR)/mm_modify_prescoef.$(OBJEXT): $(SRCDIR)/mm_modify_prescoef_fort.h
$(SRCDIR)/mmbcvelocity.$(OBJEXT): $(SRCDIR)/mmbcvelocity_fort.h
$(SRCDIR)/mmcelltypeinit.$(OBJEXT): $(SRCDIR)/mmcelltypeinit_fort.h
$(SRCDIR)/mmenthalpywallbc.$(OBJEXT): $(SRCDIR)/mmenthalpywallbc_fort.h
$(SRCDIR)/mmmomsrc.$(OBJEXT): $(SRCDIR)/mmmomsrc_fort.h
$(SRCDIR)/mmscalarwallbc.$(OBJEXT): $(SRCDIR)/mmscalarwallbc_fort.h
$(SRCDIR)/mmwallbc.$(OBJEXT): $(SRCDIR)/mmwallbc_fort.h
$(SRCDIR)/mmwallbc_trans.$(OBJEXT): $(SRCDIR)/mmwallbc_trans_fort.h
$(SRCDIR)/normpress.$(OBJEXT): $(SRCDIR)/normpress_fort.h
$(SRCDIR)/prescoef.$(OBJEXT): $(SRCDIR)/prescoef_fort.h
$(SRCDIR)/prescoef_var.$(OBJEXT): $(SRCDIR)/prescoef_var_fort.h
$(SRCDIR)/pressrcpred.$(OBJEXT): $(SRCDIR)/pressrcpred_fort.h
$(SRCDIR)/pressrcpred_var.$(OBJEXT): $(SRCDIR)/pressrcpred_var_fort.h
$(SRCDIR)/profscalar.$(OBJEXT): $(SRCDIR)/profscalar_fort.h
$(SRCDIR)/profv.$(OBJEXT): $(SRCDIR)/profv_fort.h $(SRCDIR)/ramping.h
$(SRCDIR)/scalarvarmodel.$(OBJEXT): $(SRCDIR)/scalarvarmodel_fort.h
$(SRCDIR)/scalcoef.$(OBJEXT): $(SRCDIR)/scalcoef_fort.h
$(SRCDIR)/scalsrc.$(OBJEXT): $(SRCDIR)/scalsrc_fort.h
$(SRCDIR)/smagmodel.$(OBJEXT): $(SRCDIR)/smagmodel_fort.h
$(SRCDIR)/uvelcoef.$(OBJEXT): $(SRCDIR)/uvelcoef_fort.h
$(SRCDIR)/uvelsrc.$(OBJEXT): $(SRCDIR)/uvelsrc_fort.h
$(SRCDIR)/vvelcoef.$(OBJEXT): $(SRCDIR)/vvelcoef_fort.h
$(SRCDIR)/vvelsrc.$(OBJEXT): $(SRCDIR)/vvelsrc_fort.h
$(SRCDIR)/wallbc.$(OBJEXT): $(SRCDIR)/wallbc_fort.h
$(SRCDIR)/wvelcoef.$(OBJEXT): $(SRCDIR)/wvelcoef_fort.h
$(SRCDIR)/wvelsrc.$(OBJEXT): $(SRCDIR)/wvelsrc_fort.h
$(SRCDIR)/inc_dynamic_1loop.$(OBJEXT): $(SRCDIR)/inc_dynamic_1loop_fort.h
$(SRCDIR)/inc_dynamic_2loop.$(OBJEXT): $(SRCDIR)/inc_dynamic_2loop_fort.h
$(SRCDIR)/inc_dynamic_3loop.$(OBJEXT): $(SRCDIR)/inc_dynamic_3loop_fort.h
$(SRCDIR)/comp_dynamic_1loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_1loop_fort.h
$(SRCDIR)/comp_dynamic_2loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_2loop_fort.h
$(SRCDIR)/comp_dynamic_3loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_3loop_fort.h
$(SRCDIR)/comp_dynamic_4loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_4loop_fort.h
$(SRCDIR)/comp_dynamic_5loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_5loop_fort.h
$(SRCDIR)/comp_dynamic_6loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_6loop_fort.h
$(SRCDIR)/comp_dynamic_7loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_7loop_fort.h
$(SRCDIR)/comp_dynamic_8loop.$(OBJEXT): $(SRCDIR)/comp_dynamic_8loop_fort.h
$(SRCDIR)/intrusion_computevel.$(OBJEXT): $(SRCDIR)/intrusion_computevel_fort.h
$(SRCDIR)/mmbcvelocity_momex.$(OBJEXT): $(SRCDIR)/mmbcvelocity_momex_fort.h
$(SRCDIR)/mmbcenthalpy_energyex.$(OBJEXT): $(SRCDIR)/mmbcenthalpy_energyex_fort.h

