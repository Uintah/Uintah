//----- TimeIntegratorLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_TimeIntegratorLabel_h
#define Uintah_Components_Arches_TimeIntegratorLabel_h

/**************************************
CLASS
   TimeIntegratorLabel
   
   Class TimeIntegratorLabel blah blah blah

GENERAL INFORMATION
   TimeIntegratorLabel.h - declaration of the class
   
   Author:
   
   Creation Date: 
   
   C-SAFE 
   
   Copyright U of U 

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

namespace Uintah {
  class ArchesLabel;

  class TimeIntegratorStepType {
    public:
    TimeIntegratorStepType();
    ~TimeIntegratorStepType();
    enum Type {FE, Predictor, Corrector, OldPredictor, OldCorrector,
	       Intermediate, CorrectorRK3};
    private:
    TimeIntegratorStepType(const TimeIntegratorStepType&);
    TimeIntegratorStepType& operator=(const TimeIntegratorStepType&);
  };

  class TimeIntegratorStepNumber {
    public:
    TimeIntegratorStepNumber();
    ~TimeIntegratorStepNumber();
    enum Number {First, Second, Third};
    private:
    TimeIntegratorStepNumber(const TimeIntegratorStepNumber&);
    TimeIntegratorStepNumber& operator=(const TimeIntegratorStepNumber&);
  };

  class TimeIntegratorLabel  {
    public:
    string integrator_step_name;
    int integrator_step_number;
    bool integrator_last_step;
    double time_multiplier;
    double factor_old;
    double factor_new;
    double factor_divide;
    const VarLabel* scalar_in;
    const VarLabel* density_in;
    const VarLabel* old_scalar_in;
    const VarLabel* old_density_in;
    const VarLabel* viscosity_in;
    const VarLabel* uvelocity_in;
    const VarLabel* vvelocity_in;
    const VarLabel* wvelocity_in;
    const VarLabel* old_uvelocity_in;
    const VarLabel* old_vvelocity_in;
    const VarLabel* old_wvelocity_in;
    const VarLabel* maxabsu_in;
    const VarLabel* maxabsv_in;
    const VarLabel* maxabsw_in;
    const VarLabel* scalar_out;
    const VarLabel* reactscalar_in;
    const VarLabel* old_reactscalar_in;
    const VarLabel* reactscalar_out;
    const VarLabel* enthalpy_in;
    const VarLabel* old_enthalpy_in;
    const VarLabel* enthalpy_out;
    const VarLabel* density_out;
    const VarLabel* ref_density;
    const VarLabel* rho2_density;
    const VarLabel* pressure_in;
    const VarLabel* uvelhat_out;
    const VarLabel* vvelhat_out;
    const VarLabel* wvelhat_out;
    const VarLabel* pressure_out;
    const VarLabel* pressure_guess;
    const VarLabel* uvelocity_out;
    const VarLabel* vvelocity_out;
    const VarLabel* wvelocity_out;
    const VarLabel* maxabsu_out;
    const VarLabel* maxabsv_out;
    const VarLabel* maxabsw_out;
    const VarLabel* tke_out;
    const VarLabel* flowIN; 
    const VarLabel* flowOUT;
    const VarLabel* denAccum;
    const VarLabel* floutbc;
    const VarLabel* areaOUT;
    const VarLabel* viscosity_out;


    TimeIntegratorLabel(const ArchesLabel* lab, 
			 TimeIntegratorStepType::Type intgType)
      {
	switch(intgType) {

	  case TimeIntegratorStepType::FE:
	    integrator_step_name = "FE";
	    integrator_step_number = TimeIntegratorStepNumber::First;
	    integrator_last_step = true;
            time_multiplier = 1.0;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    scalar_in = lab->d_scalarOUTBCLabel;
	    density_in = lab->d_densityINLabel;
	    old_scalar_in = lab->d_scalarOUTBCLabel;
	    old_density_in = lab->d_densityINLabel;
	    viscosity_in = lab->d_viscosityINLabel;
	    uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    old_uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    old_vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    old_wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    scalar_out = lab->d_scalarSPLabel;
	    reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    old_reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    reactscalar_out = lab->d_reactscalarSPLabel;
	    enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    old_enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    enthalpy_out = lab->d_enthalpySPLabel;
	    density_out = lab->d_densityCPLabel;
	    ref_density = lab->d_refDensity_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePSLabel;
	    uvelhat_out = lab->d_uVelRhoHatLabel;
	    vvelhat_out = lab->d_vVelRhoHatLabel;
	    wvelhat_out = lab->d_wVelRhoHatLabel;
	    pressure_out = lab->d_pressureSPBCLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    uvelocity_out = lab->d_uVelocitySPBCLabel;
	    vvelocity_out = lab->d_vVelocitySPBCLabel;
	    wvelocity_out = lab->d_wVelocitySPBCLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    viscosity_out = lab->d_viscosityCTSLabel;
	  break;

	  case TimeIntegratorStepType::Predictor:
	    integrator_step_name = "Predictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
	    integrator_last_step = false;
            time_multiplier = 1.0;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    scalar_in = lab->d_scalarOUTBCLabel;
	    density_in = lab->d_densityINLabel;
	    old_scalar_in = lab->d_scalarOUTBCLabel;
	    old_density_in = lab->d_densityINLabel;
	    viscosity_in = lab->d_viscosityINLabel;
	    uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    old_uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    old_vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    old_wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    scalar_out = lab->d_scalarPredLabel;
	    reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    old_reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    reactscalar_out = lab->d_reactscalarPredLabel;
	    enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    old_enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    enthalpy_out = lab->d_enthalpyPredLabel;
	    density_out = lab->d_densityPredLabel;
	    ref_density = lab->d_refDensityPred_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePSLabel;
	    uvelhat_out = lab->d_uVelRhoHatLabel;
	    vvelhat_out = lab->d_vVelRhoHatLabel;
	    wvelhat_out = lab->d_wVelRhoHatLabel;
	    pressure_out = lab->d_pressurePredLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    uvelocity_out = lab->d_uVelocityPredLabel;
	    vvelocity_out = lab->d_vVelocityPredLabel;
	    wvelocity_out = lab->d_wVelocityPredLabel;
	    maxabsu_out = lab->d_maxAbsUPred_label;
	    maxabsv_out = lab->d_maxAbsVPred_label;
	    maxabsw_out = lab->d_maxAbsWPred_label;
	    tke_out = lab->d_totalKineticEnergyPredLabel;
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	    viscosity_out = lab->d_viscosityPredLabel;
	  break;

	  case TimeIntegratorStepType::OldPredictor:
	    integrator_step_name = "OldPredictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
	    integrator_last_step = false;
            time_multiplier = 0.5;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    scalar_in = lab->d_scalarOUTBCLabel;
	    density_in = lab->d_densityINLabel;
	    old_scalar_in = lab->d_scalarOUTBCLabel;
	    old_density_in = lab->d_densityINLabel;
	    viscosity_in = lab->d_viscosityINLabel;
	    uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    old_uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    old_vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    old_wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    scalar_out = lab->d_scalarPredLabel;
	    reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    old_reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    reactscalar_out = lab->d_reactscalarPredLabel;
	    enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    old_enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    enthalpy_out = lab->d_enthalpyPredLabel;
	    density_out = lab->d_densityPredLabel;
	    ref_density = lab->d_refDensityPred_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePSLabel;
	    uvelhat_out = lab->d_uVelRhoHatLabel;
	    vvelhat_out = lab->d_vVelRhoHatLabel;
	    wvelhat_out = lab->d_wVelRhoHatLabel;
	    pressure_out = lab->d_pressurePredLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    uvelocity_out = lab->d_uVelocityPredLabel;
	    vvelocity_out = lab->d_vVelocityPredLabel;
	    wvelocity_out = lab->d_wVelocityPredLabel;
	    maxabsu_out = lab->d_maxAbsUPred_label;
	    maxabsv_out = lab->d_maxAbsVPred_label;
	    maxabsw_out = lab->d_maxAbsWPred_label;
	    tke_out = lab->d_totalKineticEnergyPredLabel;
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	    viscosity_out = lab->d_viscosityPredLabel;
	  break;

	  case TimeIntegratorStepType::Corrector:
	    integrator_step_name = "Corrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
	    integrator_last_step = true;
            time_multiplier = 1.0;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 2.0;
	    scalar_in = lab->d_scalarPredLabel;
	    density_in = lab->d_densityPredLabel;
	    old_scalar_in = lab->d_scalarPredLabel;
	    old_density_in = lab->d_densityPredLabel;
	    viscosity_in = lab->d_viscosityPredLabel;
	    uvelocity_in = lab->d_uVelocityPredLabel;
	    vvelocity_in = lab->d_vVelocityPredLabel;
	    wvelocity_in = lab->d_wVelocityPredLabel;
	    old_uvelocity_in = lab->d_uVelocityPredLabel;
	    old_vvelocity_in = lab->d_vVelocityPredLabel;
	    old_wvelocity_in = lab->d_wVelocityPredLabel;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    scalar_out = lab->d_scalarSPLabel;
	    reactscalar_in = lab->d_reactscalarPredLabel;
	    old_reactscalar_in = lab->d_reactscalarPredLabel;
	    reactscalar_out = lab->d_reactscalarSPLabel;
	    enthalpy_in = lab->d_enthalpyPredLabel;
	    old_enthalpy_in = lab->d_enthalpyPredLabel;
	    enthalpy_out = lab->d_enthalpySPLabel;
	    density_out = lab->d_densityCPLabel;
	    ref_density = lab->d_refDensity_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePredLabel;
	    uvelhat_out = lab->d_uVelRhoHatCorrLabel;
	    vvelhat_out = lab->d_vVelRhoHatCorrLabel;
	    wvelhat_out = lab->d_wVelRhoHatCorrLabel;
	    pressure_out = lab->d_pressureSPBCLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    uvelocity_out = lab->d_uVelocitySPBCLabel;
	    vvelocity_out = lab->d_vVelocitySPBCLabel;
	    wvelocity_out = lab->d_wVelocitySPBCLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    viscosity_out = lab->d_viscosityCTSLabel;
	  break;

	  case TimeIntegratorStepType::OldCorrector:
	    integrator_step_name = "OldCorrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
	    integrator_last_step = true;
            time_multiplier = 1.0;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    scalar_in = lab->d_scalarPredLabel;
	    density_in = lab->d_densityPredLabel;
	    old_scalar_in = lab->d_scalarOUTBCLabel;
	    old_density_in = lab->d_densityINLabel;
	    viscosity_in = lab->d_viscosityPredLabel;
	    uvelocity_in = lab->d_uVelocityPredLabel;
	    vvelocity_in = lab->d_vVelocityPredLabel;
	    wvelocity_in = lab->d_wVelocityPredLabel;
	    old_uvelocity_in = lab->d_uVelocityOUTBCLabel;
	    old_vvelocity_in = lab->d_vVelocityOUTBCLabel;
	    old_wvelocity_in = lab->d_wVelocityOUTBCLabel;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    scalar_out = lab->d_scalarSPLabel;
	    reactscalar_in = lab->d_reactscalarPredLabel;
	    old_reactscalar_in = lab->d_reactscalarOUTBCLabel;
	    reactscalar_out = lab->d_reactscalarSPLabel;
	    enthalpy_in = lab->d_enthalpyPredLabel;
	    old_enthalpy_in = lab->d_enthalpyOUTBCLabel;
	    enthalpy_out = lab->d_enthalpySPLabel;
	    density_out = lab->d_densityCPLabel;
	    ref_density = lab->d_refDensity_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePredLabel;
	    uvelhat_out = lab->d_uVelRhoHatCorrLabel;
	    vvelhat_out = lab->d_vVelRhoHatCorrLabel;
	    wvelhat_out = lab->d_wVelRhoHatCorrLabel;
	    pressure_out = lab->d_pressureSPBCLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    uvelocity_out = lab->d_uVelocitySPBCLabel;
	    vvelocity_out = lab->d_vVelocitySPBCLabel;
	    wvelocity_out = lab->d_wVelocitySPBCLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    viscosity_out = lab->d_viscosityCTSLabel;
	  break;

	  case TimeIntegratorStepType::CorrectorRK3:
	    integrator_step_name = "CorrectorRK3";
	    integrator_step_number = TimeIntegratorStepNumber::Third;
	    integrator_last_step = true;
            time_multiplier = 1.0;
	    factor_old = 1.0;
	    factor_new = 2.0;
	    factor_divide = 3.0;
	    scalar_in = lab->d_scalarIntermLabel;
	    density_in = lab->d_densityIntermLabel;
	    old_scalar_in = lab->d_scalarIntermLabel;
	    old_density_in = lab->d_densityIntermLabel;
	    viscosity_in = lab->d_viscosityIntermLabel;
	    uvelocity_in = lab->d_uVelocityIntermLabel;
	    vvelocity_in = lab->d_vVelocityIntermLabel;
	    wvelocity_in = lab->d_wVelocityIntermLabel;
	    old_uvelocity_in = lab->d_uVelocityIntermLabel;
	    old_vvelocity_in = lab->d_vVelocityIntermLabel;
	    old_wvelocity_in = lab->d_wVelocityIntermLabel;
	    maxabsu_in = lab->d_maxAbsUInterm_label;
	    maxabsv_in = lab->d_maxAbsVInterm_label;
	    maxabsw_in = lab->d_maxAbsWInterm_label;
	    scalar_out = lab->d_scalarSPLabel;
	    reactscalar_in = lab->d_reactscalarIntermLabel;
	    old_reactscalar_in = lab->d_reactscalarIntermLabel;
	    reactscalar_out = lab->d_reactscalarSPLabel;
	    enthalpy_in = lab->d_enthalpyIntermLabel;
	    old_enthalpy_in = lab->d_enthalpyIntermLabel;
	    enthalpy_out = lab->d_enthalpySPLabel;
	    density_out = lab->d_densityCPLabel;
	    ref_density = lab->d_refDensity_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressureIntermLabel;
	    uvelhat_out = lab->d_uVelRhoHatCorrLabel;
	    vvelhat_out = lab->d_vVelRhoHatCorrLabel;
	    wvelhat_out = lab->d_wVelRhoHatCorrLabel;
	    pressure_out = lab->d_pressureSPBCLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    uvelocity_out = lab->d_uVelocitySPBCLabel;
	    vvelocity_out = lab->d_vVelocitySPBCLabel;
	    wvelocity_out = lab->d_wVelocitySPBCLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    viscosity_out = lab->d_viscosityCTSLabel;
	  break;

	  case TimeIntegratorStepType::Intermediate:
	    integrator_step_name = "Intermediate";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
	    integrator_last_step = false;
            time_multiplier = 1.0;
	    factor_old = 3.0;
	    factor_new = 1.0;
	    factor_divide = 4.0;
	    scalar_in = lab->d_scalarPredLabel;
	    density_in = lab->d_densityPredLabel;
	    old_scalar_in = lab->d_scalarPredLabel;
	    old_density_in = lab->d_densityPredLabel;
	    viscosity_in = lab->d_viscosityPredLabel;
	    uvelocity_in = lab->d_uVelocityPredLabel;
	    vvelocity_in = lab->d_vVelocityPredLabel;
	    wvelocity_in = lab->d_wVelocityPredLabel;
	    old_uvelocity_in = lab->d_uVelocityPredLabel;
	    old_vvelocity_in = lab->d_vVelocityPredLabel;
	    old_wvelocity_in = lab->d_wVelocityPredLabel;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    scalar_out = lab->d_scalarIntermLabel;
	    reactscalar_in = lab->d_reactscalarPredLabel;
	    old_reactscalar_in = lab->d_reactscalarPredLabel;
	    reactscalar_out = lab->d_reactscalarIntermLabel;
	    enthalpy_in = lab->d_enthalpyPredLabel;
	    old_enthalpy_in = lab->d_enthalpyPredLabel;
	    enthalpy_out = lab->d_enthalpyIntermLabel;
	    density_out = lab->d_densityIntermLabel;
	    ref_density = lab->d_refDensityInterm_label;
	    rho2_density = lab->d_densityPredLabel;
	    pressure_in = lab->d_pressurePredLabel;
	    uvelhat_out = lab->d_uVelRhoHatIntermLabel;
	    vvelhat_out = lab->d_vVelRhoHatIntermLabel;
	    wvelhat_out = lab->d_wVelRhoHatIntermLabel;
	    pressure_out = lab->d_pressureIntermLabel;
	    pressure_guess = lab->d_pressureIntermLabel;
	    uvelocity_out = lab->d_uVelocityIntermLabel;
	    vvelocity_out = lab->d_vVelocityIntermLabel;
	    wvelocity_out = lab->d_wVelocityIntermLabel;
	    maxabsu_out = lab->d_maxAbsUInterm_label;
	    maxabsv_out = lab->d_maxAbsVInterm_label;
	    maxabsw_out = lab->d_maxAbsWInterm_label;
	    tke_out = lab->d_totalKineticEnergyIntermLabel;
	    flowIN = lab->d_totalflowINIntermLabel;
	    flowOUT = lab->d_totalflowOUTIntermLabel;
	    denAccum = lab->d_denAccumIntermLabel;
	    floutbc = lab->d_netflowOUTBCIntermLabel;
	    areaOUT = lab->d_totalAreaOUTIntermLabel;
	    viscosity_out = lab->d_viscosityIntermLabel;
	  break;

	  default: 
		throw InvalidValue("Unknown explicit time integrator type");
	}
      }; 

    ~TimeIntegratorLabel() {};

    private:
    TimeIntegratorLabel(const TimeIntegratorLabel&);
    TimeIntegratorLabel& operator=(const TimeIntegratorLabel&);
  };
} // End namespace Uintah

#endif
