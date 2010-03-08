/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

#include <Core/Exceptions/InvalidValue.h>

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
    const VarLabel* ref_density;
    const VarLabel* pressure_out;
    const VarLabel* pressure_guess;
    const VarLabel* tke_out;
    const VarLabel* flowIN; 
    const VarLabel* flowOUT;
    const VarLabel* denAccum;
    const VarLabel* floutbc;
    const VarLabel* areaOUT;
    const VarLabel* negativeDensityGuess;
    const VarLabel* negativeEKTDensityGuess;
    const VarLabel* densityLag;
    const VarLabel* ummsLnError;
    const VarLabel* vmmsLnError;
    const VarLabel* wmmsLnError;
    const VarLabel* smmsLnError;
    const VarLabel* gradpmmsLnError;
    const VarLabel* ummsExactSol;
    const VarLabel* vmmsExactSol;
    const VarLabel* wmmsExactSol;
    const VarLabel* smmsExactSol;
    const VarLabel* gradpmmsExactSol;


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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressurePSLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressurePSLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
            ref_density = lab->d_refDensityPred_label;
            pressure_out = lab->d_pressurePredLabel;
            pressure_guess = lab->d_pressurePSLabel;
            tke_out = lab->d_totalKineticEnergyPredLabel;
            flowIN = lab->d_totalflowINPredLabel;
            flowOUT = lab->d_totalflowOUTPredLabel;
            denAccum = lab->d_denAccumPredLabel;
            floutbc = lab->d_netflowOUTBCPredLabel;
            areaOUT = lab->d_totalAreaOUTPredLabel;
            negativeDensityGuess = lab->d_negativeDensityGuessPred_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuessPred_label;
            densityLag = lab->d_densityLagPred_label;
            ummsLnError = lab->d_totalummsLnErrorPredLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorPredLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorPredLabel;
            smmsLnError = lab->d_totalsmmsLnErrorPredLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorPredLabel;
            ummsExactSol = lab->d_totalummsExactSolPredLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolPredLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolPredLabel;
            smmsExactSol = lab->d_totalsmmsExactSolPredLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolPredLabel;
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
            ref_density = lab->d_refDensityPred_label;
            pressure_out = lab->d_pressurePredLabel;
            pressure_guess = lab->d_pressurePSLabel;
            tke_out = lab->d_totalKineticEnergyPredLabel;
            flowIN = lab->d_totalflowINPredLabel;
            flowOUT = lab->d_totalflowOUTPredLabel;
            denAccum = lab->d_denAccumPredLabel;
            floutbc = lab->d_netflowOUTBCPredLabel;
            areaOUT = lab->d_totalAreaOUTPredLabel;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuessPred_label;
            densityLag = lab->d_densityLagPred_label;
            ummsLnError = lab->d_totalummsLnErrorPredLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorPredLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorPredLabel;
            smmsLnError = lab->d_totalsmmsLnErrorPredLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorPredLabel;
            ummsExactSol = lab->d_totalummsExactSolPredLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolPredLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolPredLabel;
            smmsExactSol = lab->d_totalsmmsExactSolPredLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolPredLabel;
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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressurePredLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressurePredLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressureIntermLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
            ref_density = lab->d_refDensityInterm_label;
            pressure_out = lab->d_pressureIntermLabel;
            pressure_guess = lab->d_pressurePredLabel;
            tke_out = lab->d_totalKineticEnergyIntermLabel;
            flowIN = lab->d_totalflowINIntermLabel;
            flowOUT = lab->d_totalflowOUTIntermLabel;
            denAccum = lab->d_denAccumIntermLabel;
            floutbc = lab->d_netflowOUTBCIntermLabel;
            areaOUT = lab->d_totalAreaOUTIntermLabel;
            negativeDensityGuess = lab->d_negativeDensityGuessInterm_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuessInterm_label;
            densityLag = lab->d_densityLagInterm_label;
            ummsLnError = lab->d_totalummsLnErrorIntermLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorIntermLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorIntermLabel;
            smmsLnError = lab->d_totalsmmsLnErrorIntermLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorIntermLabel;
            ummsExactSol = lab->d_totalummsExactSolIntermLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolIntermLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolIntermLabel;
            smmsExactSol = lab->d_totalsmmsExactSolIntermLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolIntermLabel;
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
            ref_density = lab->d_refDensityPred_label;
            pressure_out = lab->d_pressurePredLabel;
            pressure_guess = lab->d_pressurePSLabel;
            tke_out = lab->d_totalKineticEnergyPredLabel;
            flowIN = lab->d_totalflowINPredLabel;
            flowOUT = lab->d_totalflowOUTPredLabel;
            denAccum = lab->d_denAccumPredLabel;
            floutbc = lab->d_netflowOUTBCPredLabel;
            areaOUT = lab->d_totalAreaOUTPredLabel;
            negativeDensityGuess = lab->d_negativeDensityGuessPred_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuessPred_label;
            densityLag = lab->d_densityLagPred_label;
            ummsLnError = lab->d_totalummsLnErrorPredLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorPredLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorPredLabel;
            smmsLnError = lab->d_totalsmmsLnErrorPredLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorPredLabel;
            ummsExactSol = lab->d_totalummsExactSolPredLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolPredLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolPredLabel;
            smmsExactSol = lab->d_totalsmmsExactSolPredLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolPredLabel;
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
            ref_density = lab->d_refDensityInterm_label;
            pressure_out = lab->d_pressureIntermLabel;
            pressure_guess = lab->d_pressurePredLabel;
            tke_out = lab->d_totalKineticEnergyIntermLabel;
            flowIN = lab->d_totalflowINIntermLabel;
            flowOUT = lab->d_totalflowOUTIntermLabel;
            denAccum = lab->d_denAccumIntermLabel;
            floutbc = lab->d_netflowOUTBCIntermLabel;
            areaOUT = lab->d_totalAreaOUTIntermLabel;
            negativeDensityGuess = lab->d_negativeDensityGuessInterm_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuessInterm_label;
            densityLag = lab->d_densityLagInterm_label;
            ummsLnError = lab->d_totalummsLnErrorIntermLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorIntermLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorIntermLabel;
            smmsLnError = lab->d_totalsmmsLnErrorIntermLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorIntermLabel;
            ummsExactSol = lab->d_totalummsExactSolIntermLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolIntermLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolIntermLabel;
            smmsExactSol = lab->d_totalsmmsExactSolIntermLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolIntermLabel;
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
            ref_density = lab->d_refDensity_label;
            pressure_out = lab->d_pressurePSLabel;
            pressure_guess = lab->d_pressureIntermLabel;
            tke_out = lab->d_totalKineticEnergyLabel;
            flowIN = lab->d_totalflowINLabel;
            flowOUT = lab->d_totalflowOUTLabel;
            denAccum = lab->d_denAccumLabel;
            floutbc = lab->d_netflowOUTBCLabel;
            areaOUT = lab->d_totalAreaOUTLabel;
            negativeDensityGuess = lab->d_negativeDensityGuess_label;
            negativeEKTDensityGuess = lab->d_negativeEKTDensityGuess_label;
            densityLag = lab->d_densityLag_label;
            ummsLnError = lab->d_totalummsLnErrorLabel;
            vmmsLnError = lab->d_totalvmmsLnErrorLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
            smmsLnError = lab->d_totalsmmsLnErrorLabel;
            gradpmmsLnError = lab->d_totalgradpmmsLnErrorLabel;
            ummsExactSol = lab->d_totalummsExactSolLabel;
            vmmsExactSol = lab->d_totalvmmsExactSolLabel;
            wmmsExactSol = lab->d_totalwmmsExactSolLabel;
            smmsExactSol = lab->d_totalsmmsExactSolLabel;
            gradpmmsExactSol = lab->d_totalgradpmmsExactSolLabel;
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
