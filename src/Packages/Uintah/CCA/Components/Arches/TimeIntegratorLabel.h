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
    const VarLabel* flowIN; 
    const VarLabel* flowOUT;
    const VarLabel* denAccum;
    const VarLabel* floutbc;
    const VarLabel* areaOUT;


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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	  break;

	  case TimeIntegratorStepType::FE:
	    integrator_step_name = "FE";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = false;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	  break;

	  case TimeIntegratorStepType::Predictor:
	    integrator_step_name = "Predictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	  break;

	  case TimeIntegratorStepType::OldPredictor:
	    integrator_step_name = "OldPredictor";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 0.5;
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
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	  break;

	  case TimeIntegratorStepType::Corrector:
	    integrator_step_name = "Corrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	  break;

	  case TimeIntegratorStepType::OldCorrector:
	    integrator_step_name = "OldCorrector";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	  break;

	  case TimeIntegratorStepType::CorrectorRK3:
	    integrator_step_name = "CorrectorRK3";
	    integrator_step_number = TimeIntegratorStepNumber::Third;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
	  break;

	  case TimeIntegratorStepType::Intermediate:
	    integrator_step_name = "Intermediate";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = false;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINIntermLabel;
	    flowOUT = lab->d_totalflowOUTIntermLabel;
	    denAccum = lab->d_denAccumIntermLabel;
	    floutbc = lab->d_netflowOUTBCIntermLabel;
	    areaOUT = lab->d_totalAreaOUTIntermLabel;
	  break;

	  case TimeIntegratorStepType::BEEmulation1:
	    integrator_step_name = "BEEmulation1";
	    integrator_step_number = TimeIntegratorStepNumber::First;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINPredLabel;
	    flowOUT = lab->d_totalflowOUTPredLabel;
	    denAccum = lab->d_denAccumPredLabel;
	    floutbc = lab->d_netflowOUTBCPredLabel;
	    areaOUT = lab->d_totalAreaOUTPredLabel;
	  break;

	  case TimeIntegratorStepType::BEEmulation2:
	    integrator_step_name = "BEEmulation2";
	    integrator_step_number = TimeIntegratorStepNumber::Second;
            multiple_steps = true;
	    integrator_last_step = false;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINIntermLabel;
	    flowOUT = lab->d_totalflowOUTIntermLabel;
	    denAccum = lab->d_denAccumIntermLabel;
	    floutbc = lab->d_netflowOUTBCIntermLabel;
	    areaOUT = lab->d_totalAreaOUTIntermLabel;
	  break;

	  case TimeIntegratorStepType::BEEmulation3:
	    integrator_step_name = "BEEmulation3";
	    integrator_step_number = TimeIntegratorStepNumber::Third;
            multiple_steps = true;
	    integrator_last_step = true;
	    use_old_values = true;
	    recursion = false;
            time_multiplier = 1.0;
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
	    flowIN = lab->d_totalflowINLabel;
	    flowOUT = lab->d_totalflowOUTLabel;
	    denAccum = lab->d_denAccumLabel;
	    floutbc = lab->d_netflowOUTBCLabel;
	    areaOUT = lab->d_totalAreaOUTLabel;
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
