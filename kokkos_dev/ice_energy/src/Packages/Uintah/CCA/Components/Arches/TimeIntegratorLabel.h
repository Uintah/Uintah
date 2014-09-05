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
    enum Type {BE, FE, Predictor, Corrector, OldPredictor, OldCorrector,
	       Intermediate, CorrectorRK3, BEEmulation1, BEEmulation2, BEEmulation3};
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
    bool multiple_steps;
    bool integrator_last_step;
    bool use_old_values;
    bool recursion;
    double time_multiplier;
    double time_position_multiplier_before_average;
    double time_position_multiplier_after_average;
    double factor_old;
    double factor_new;
    double factor_divide;
    const VarLabel* maxabsu_in;
    const VarLabel* maxabsv_in;
    const VarLabel* maxabsw_in;
    const VarLabel* ref_density;
    const VarLabel* pressure_out;
    const VarLabel* pressure_guess;
    const VarLabel* maxabsu_out;
    const VarLabel* maxabsv_out;
    const VarLabel* maxabsw_out;
    const VarLabel* tke_out;
    const VarLabel* sumSijSij_out;
    const VarLabel* sumUUU_out;
    const VarLabel* sumDll_out;
    const VarLabel* sumDllMinus_out;
    const VarLabel* sumPoints_out;
    const VarLabel* flowIN; 
    const VarLabel* flowOUT;
    const VarLabel* denAccum;
    const VarLabel* floutbc;
    const VarLabel* areaOUT;
    const VarLabel* maxuxplus_in;
    const VarLabel* maxuxplus_out;
    const VarLabel* avuxplus_in;
    const VarLabel* avuxplus_out;


    TimeIntegratorLabel(const ArchesLabel* lab, 
			 TimeIntegratorStepType::Type intgType)
      {
	switch(intgType) {

	  case TimeIntegratorStepType::BE:
	    integrator_step_name = "BE";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = false;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = true;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplus_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplus_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  case TimeIntegratorStepType::FE:
	    integrator_step_name = "FE";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = false;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplus_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplus_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  case TimeIntegratorStepType::Predictor:
	    integrator_step_name = "Predictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    ref_density = lab->d_refDensityPred_label;
	    pressure_out = lab->d_pressurePredLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    maxabsu_out = lab->d_maxAbsUPred_label;
	    maxabsv_out = lab->d_maxAbsVPred_label;
	    maxabsw_out = lab->d_maxAbsWPred_label;
	    tke_out = lab->d_totalKineticEnergyPredLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	    maxuxplus_in = lab->d_maxUxplus_label;
	    maxuxplus_out = lab->d_maxUxplusPred_label;
	    avuxplus_in = lab->d_avUxplus_label;
	    avuxplus_out = lab->d_avUxplusPred_label;
	  break;

	  case TimeIntegratorStepType::OldPredictor:
	    integrator_step_name = "OldPredictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 0.5;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    ref_density = lab->d_refDensityPred_label;
	    pressure_out = lab->d_pressurePredLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    maxabsu_out = lab->d_maxAbsUPred_label;
	    maxabsv_out = lab->d_maxAbsVPred_label;
	    maxabsw_out = lab->d_maxAbsWPred_label;
	    tke_out = lab->d_totalKineticEnergyPredLabel;
	    sumSijSij_out=lab->d_sumSijSijPredLabel;
	    sumUUU_out=lab->d_sumUUUPredLabel;
	    sumDll_out=lab->d_sumDllPredLabel;
	    sumDllMinus_out=lab->d_sumDllMinusPredLabel;
	    sumPoints_out=lab->d_sumPointsPredLabel;
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	    maxuxplus_in = lab->d_maxUxplus_label;
	    maxuxplus_out = lab->d_maxUxplusPred_label;
	    avuxplus_in = lab->d_avUxplus_label;
	    avuxplus_out = lab->d_avUxplusPred_label;
	  break;

	  case TimeIntegratorStepType::Corrector:
	    integrator_step_name = "Corrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 2.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 1.0;
	    factor_new = 1.0;
	    factor_divide = 2.0;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplusPred_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplusPred_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  case TimeIntegratorStepType::OldCorrector:
	    integrator_step_name = "OldCorrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplusPred_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplusPred_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  case TimeIntegratorStepType::CorrectorRK3:
	    integrator_step_name = "CorrectorRK3";
	    integrator_step_number = TimeIntegratorStepNumber::Third;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.5;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 1.0;
	    factor_new = 2.0;
	    factor_divide = 3.0;
	    maxabsu_in = lab->d_maxAbsUInterm_label;
	    maxabsv_in = lab->d_maxAbsVInterm_label;
	    maxabsw_in = lab->d_maxAbsWInterm_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressureIntermLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplusInterm_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplusInterm_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  case TimeIntegratorStepType::Intermediate:
	    integrator_step_name = "Intermediate";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 2.0;
            time_position_multiplier_after_average = 0.5;
	    factor_old = 3.0;
	    factor_new = 1.0;
	    factor_divide = 4.0;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    ref_density = lab->d_refDensityInterm_label;
	    pressure_out = lab->d_pressureIntermLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    maxabsu_out = lab->d_maxAbsUInterm_label;
	    maxabsv_out = lab->d_maxAbsVInterm_label;
	    maxabsw_out = lab->d_maxAbsWInterm_label;
	    tke_out = lab->d_totalKineticEnergyIntermLabel;
	    sumSijSij_out=lab->d_sumSijSijIntermLabel;
	    sumUUU_out=lab->d_sumUUUIntermLabel;
	    sumDll_out=lab->d_sumDllIntermLabel;
	    sumDllMinus_out=lab->d_sumDllMinusIntermLabel;
	    sumPoints_out=lab->d_sumPointsIntermLabel;
	    flowIN = lab->d_totalflowINIntermLabel;
	    flowOUT = lab->d_totalflowOUTIntermLabel;
	    denAccum = lab->d_denAccumIntermLabel;
	    floutbc = lab->d_netflowOUTBCIntermLabel;
	    areaOUT = lab->d_totalAreaOUTIntermLabel;
	    maxuxplus_in = lab->d_maxUxplusPred_label;
	    maxuxplus_out = lab->d_maxUxplusInterm_label;
	    avuxplus_in = lab->d_avUxplusPred_label;
	    avuxplus_out = lab->d_avUxplusInterm_label;
	  break;

	  case TimeIntegratorStepType::BEEmulation1:
	    integrator_step_name = "BEEmulation1";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsU_label;
	    maxabsv_in = lab->d_maxAbsV_label;
	    maxabsw_in = lab->d_maxAbsW_label;
	    ref_density = lab->d_refDensityPred_label;
	    pressure_out = lab->d_pressurePredLabel;
	    pressure_guess = lab->d_pressurePSLabel;
	    maxabsu_out = lab->d_maxAbsUPred_label;
	    maxabsv_out = lab->d_maxAbsVPred_label;
	    maxabsw_out = lab->d_maxAbsWPred_label;
	    tke_out = lab->d_totalKineticEnergyPredLabel;
	    sumSijSij_out=lab->d_sumSijSijPredLabel;
	    sumUUU_out=lab->d_sumUUUPredLabel;
	    sumDll_out=lab->d_sumDllPredLabel;
	    sumDllMinus_out=lab->d_sumDllMinusPredLabel;
	    sumPoints_out=lab->d_sumPointsPredLabel;
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	    maxuxplus_in = lab->d_maxUxplus_label;
	    maxuxplus_out = lab->d_maxUxplusPred_label;
	    avuxplus_in = lab->d_avUxplus_label;
	    avuxplus_out = lab->d_avUxplusPred_label;
	  break;

	  case TimeIntegratorStepType::BEEmulation2:
	    integrator_step_name = "BEEmulation2";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsUPred_label;
	    maxabsv_in = lab->d_maxAbsVPred_label;
	    maxabsw_in = lab->d_maxAbsWPred_label;
	    ref_density = lab->d_refDensityInterm_label;
	    pressure_out = lab->d_pressureIntermLabel;
	    pressure_guess = lab->d_pressurePredLabel;
	    maxabsu_out = lab->d_maxAbsUInterm_label;
	    maxabsv_out = lab->d_maxAbsVInterm_label;
	    maxabsw_out = lab->d_maxAbsWInterm_label;
	    tke_out = lab->d_totalKineticEnergyIntermLabel;
	    sumSijSij_out=lab->d_sumSijSijIntermLabel;
	    sumUUU_out=lab->d_sumUUUIntermLabel;
	    sumDll_out=lab->d_sumDllIntermLabel;
	    sumDllMinus_out=lab->d_sumDllMinusIntermLabel;
	    sumPoints_out=lab->d_sumPointsIntermLabel;
	    flowIN = lab->d_totalflowINIntermLabel;
	    flowOUT = lab->d_totalflowOUTIntermLabel;
	    denAccum = lab->d_denAccumIntermLabel;
	    floutbc = lab->d_netflowOUTBCIntermLabel;
	    areaOUT = lab->d_totalAreaOUTIntermLabel;
	    maxuxplus_in = lab->d_maxUxplusPred_label;
	    maxuxplus_out = lab->d_maxUxplusInterm_label;
	    avuxplus_in = lab->d_avUxplusPred_label;
	    avuxplus_out = lab->d_avUxplusInterm_label;
	  break;

	  case TimeIntegratorStepType::BEEmulation3:
	    integrator_step_name = "BEEmulation3";
	    integrator_step_number = TimeIntegratorStepNumber::Third;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_position_multiplier_after_average = 1.0;
	    factor_old = 0.0;
	    factor_new = 1.0;
	    factor_divide = 1.0;
	    maxabsu_in = lab->d_maxAbsUInterm_label;
	    maxabsv_in = lab->d_maxAbsVInterm_label;
	    maxabsw_in = lab->d_maxAbsWInterm_label;
	    ref_density = lab->d_refDensity_label;
	    pressure_out = lab->d_pressurePSLabel;
	    pressure_guess = lab->d_pressureIntermLabel;
	    maxabsu_out = lab->d_maxAbsU_label;
	    maxabsv_out = lab->d_maxAbsV_label;
	    maxabsw_out = lab->d_maxAbsW_label;
	    tke_out = lab->d_totalKineticEnergyLabel;
	    sumSijSij_out=lab->d_sumSijSijLabel;
	    sumUUU_out=lab->d_sumUUULabel;
	    sumDll_out=lab->d_sumDllLabel;
	    sumDllMinus_out=lab->d_sumDllMinusLabel;
	    sumPoints_out=lab->d_sumPointsLabel;
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	    maxuxplus_in = lab->d_maxUxplusInterm_label;
	    maxuxplus_out = lab->d_maxUxplus_label;
	    avuxplus_in = lab->d_avUxplusInterm_label;
	    avuxplus_out = lab->d_avUxplus_label;
	  break;

	  default: 
		throw InvalidValue("Unknown explicit time integrator type", __FILE__, __LINE__);
	}
      }; 

    ~TimeIntegratorLabel() {};

    private:
    TimeIntegratorLabel(const TimeIntegratorLabel&);
    TimeIntegratorLabel& operator=(const TimeIntegratorLabel&);
  };
} // End namespace Uintah

#endif
