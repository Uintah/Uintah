/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
   
   
KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/InternalError.h>

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
    
    // PROGRAMMER:  if you added a variable
    // make sure you initialize it in initializeVars()
    // and catch it in catchUninitializedVars ()
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
    const VarLabel* ref_pressure;
    const VarLabel* pressure_out;
    const VarLabel* pressure_guess;
    const VarLabel* tke_out;
    const VarLabel* flowIN; 
    const VarLabel* flowOUT;
    const VarLabel* denAccum;
    const VarLabel* floutbc;
    const VarLabel* areaOUT;
    const VarLabel* negativeDensityGuess;
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

    //__________________________________
    //  initialize all the variabless
    void initializeVars(){
      areaOUT                  =  NULL;
      denAccum                 =  NULL;
      densityLag               =  NULL;
      factor_divide            = -9;                 
      factor_new               = -9;                 
      factor_old               = -9;                 
      floutbc                  =  NULL;
      flowIN                   =  NULL;
      flowOUT                  =  NULL;
      gradpmmsExactSol         =  NULL;
      gradpmmsLnError          =  NULL;
      integrator_step_name     = "-9";
      integrator_step_number   =  -9;
      negativeDensityGuess     =  NULL;
      pressure_guess           =  NULL;
      pressure_out             =  NULL;
      ref_density              =  NULL;
      smmsExactSol             =  NULL;
      smmsLnError              =  NULL;
      time_multiplier          = -9;
      time_position_multiplier_after_average    = -9;
      time_position_multiplier_before_average   = -9;
      tke_out                  =  NULL;
      ummsExactSol             =  NULL;
      ummsLnError              =  NULL;
      vmmsExactSol             =  NULL;
      vmmsLnError              =  NULL;
      wmmsExactSol             =  NULL;
      wmmsLnError              =  NULL;
    }
    
    //__________________________________
    //  Test for uninitialized Variables
    void catchUninitializedVars(){
      bool test =( (areaOUT      ==  NULL ) ||
      ( denAccum                 ==  NULL ) ||
      ( densityLag               ==  NULL ) ||
      ( factor_divide            == -9 )    ||              
      ( factor_new               == -9 )    ||              
      ( factor_old               == -9 )    ||              
      ( floutbc                  ==  NULL ) ||
      ( flowIN                   ==  NULL ) ||
      ( flowOUT                  ==  NULL ) ||
      ( gradpmmsExactSol         ==  NULL ) ||
      ( gradpmmsLnError          ==  NULL ) ||
      ( integrator_step_name     == "-9" )  ||
      ( integrator_step_number   ==  -9 )   ||
      ( negativeDensityGuess     ==  NULL ) ||
      ( pressure_guess           ==  NULL ) ||
      ( pressure_out             ==  NULL ) ||
      ( ref_density              ==  NULL ) ||
      ( smmsExactSol             ==  NULL ) ||
      ( smmsLnError              ==  NULL ) ||
      ( time_multiplier          == -9 )    ||
      ( time_position_multiplier_after_average    == -9 ) ||
      ( time_position_multiplier_before_average   == -9 ) ||
      ( tke_out                  ==  NULL ) ||
      ( ummsExactSol             ==  NULL ) ||
      ( ummsLnError              ==  NULL ) ||
      ( vmmsExactSol             ==  NULL ) ||
      ( vmmsLnError              ==  NULL ) ||
      ( wmmsExactSol             ==  NULL ) ||
      ( wmmsLnError              ==  NULL ) );
      
      if (test){
        throw InternalError("ERROR: unintialized variable in TimeIntegrator.h", __FILE__, __LINE__);
      }
    }

    //______________________________________________________________________
    //
    TimeIntegratorLabel(const ArchesLabel* lab, 
                         TimeIntegratorStepType::Type intgType)
      {
      
        initializeVars();
        
        switch(intgType) {

          case TimeIntegratorStepType::BE:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "BE";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  false;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  true;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_multiplier          =  1.0;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
          break;

          case TimeIntegratorStepType::FE:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "FE";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  false;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorLabel;
          break;

          case TimeIntegratorStepType::Predictor:
            areaOUT                  =  lab->d_totalAreaOUTPredLabel;
            denAccum                 =  lab->d_denAccumPredLabel;
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCPredLabel;
            flowIN                   =  lab->d_totalflowINPredLabel;
            flowOUT                  =  lab->d_totalflowOUTPredLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolPredLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorPredLabel;
            integrator_last_step     =  false;
            integrator_step_name     =  "Predictor";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuessPred_label;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePredLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityPred_label;
            ref_pressure             =  lab->d_refPressurePred_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolPredLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorPredLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyPredLabel;
            ummsExactSol             =  lab->d_totalummsExactSolPredLabel;
            ummsLnError              =  lab->d_totalummsLnErrorPredLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolPredLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorPredLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolPredLabel;
            wmmsLnError              = lab->d_totalwmmsLnErrorPredLabel;
          break;

          case TimeIntegratorStepType::OldPredictor:
            areaOUT                  =  lab->d_totalAreaOUTPredLabel;
            denAccum                 =  lab->d_denAccumPredLabel;
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCPredLabel;
            flowIN                   =  lab->d_totalflowINPredLabel;
            flowOUT                  =  lab->d_totalflowOUTPredLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolPredLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorPredLabel;
            integrator_last_step     =  false;
            integrator_step_name     =  "OldPredictor";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  true;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePredLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityPred_label;
            ref_pressure             =  lab->d_refPressurePred_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolPredLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorPredLabel;
            time_multiplier          =  0.5;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyPredLabel;
            ummsExactSol             =  lab->d_totalummsExactSolPredLabel;
            ummsLnError              =  lab->d_totalummsLnErrorPredLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolPredLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorPredLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolPredLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorPredLabel;
          break;

          case TimeIntegratorStepType::Corrector:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  2.0;
            factor_new               =  1.0;
            factor_old               =  1.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "Corrector";
            integrator_step_number   =  TimeIntegratorStepNumber::Second;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressurePredLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 2.0;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  false;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
            wmmsLnError = lab->d_totalwmmsLnErrorLabel;
          break;

          case TimeIntegratorStepType::OldCorrector:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "OldCorrector";
            integrator_step_number   =  TimeIntegratorStepNumber::Second;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressurePredLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorLabel;
          break;

          case TimeIntegratorStepType::CorrectorRK3:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  3.0;
            factor_new               =  2.0;
            factor_old               =  1.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "CorrectorRK3";
            integrator_step_number   =  TimeIntegratorStepNumber::Third;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressureIntermLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.5;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  false;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorLabel;
          break;

          case TimeIntegratorStepType::Intermediate:
            areaOUT                  =  lab->d_totalAreaOUTIntermLabel;
            denAccum                 =  lab->d_denAccumIntermLabel;
            densityLag               =  lab->d_densityLagInterm_label;
            factor_divide            =  4.0;
            factor_new               =  1.0;
            factor_old               =  3.0;
            floutbc                  =  lab->d_netflowOUTBCIntermLabel;
            flowIN                   =  lab->d_totalflowINIntermLabel;
            flowOUT                  =  lab->d_totalflowOUTIntermLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolIntermLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorIntermLabel;
            integrator_last_step     =  false;
            integrator_step_name     =  "Intermediate";
            integrator_step_number   =  TimeIntegratorStepNumber::Second;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuessInterm_label;
            pressure_guess           =  lab->d_pressurePredLabel;
            pressure_out             =  lab->d_pressureIntermLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityInterm_label;
            ref_pressure             =  lab->d_refPressureInterm_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolIntermLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorIntermLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 0.5;
            time_position_multiplier_before_average = 2.0;
            tke_out                  =  lab->d_totalKineticEnergyIntermLabel;
            ummsExactSol             =  lab->d_totalummsExactSolIntermLabel;
            ummsLnError              =  lab->d_totalummsLnErrorIntermLabel;
            use_old_values           =  false;
            vmmsExactSol             =  lab->d_totalvmmsExactSolIntermLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorIntermLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolIntermLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorIntermLabel;
          break;

          case TimeIntegratorStepType::BEEmulation1:
            areaOUT                  =  lab->d_totalAreaOUTPredLabel;
            denAccum                 =  lab->d_denAccumPredLabel;
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCPredLabel;
            flowIN                   =  lab->d_totalflowINPredLabel;
            flowOUT                  =  lab->d_totalflowOUTPredLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolPredLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorPredLabel;
            integrator_last_step     =  false;
            integrator_step_name     =  "BEEmulation1";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuessPred_label;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePredLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityPred_label;
            ref_pressure             =  lab->d_refPressurePred_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolPredLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorPredLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyPredLabel;
            ummsExactSol             =  lab->d_totalummsExactSolPredLabel;
            ummsLnError              =  lab->d_totalummsLnErrorPredLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolPredLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorPredLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolPredLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorPredLabel;
          break;

          case TimeIntegratorStepType::BEEmulation2:
            areaOUT                  =  lab->d_totalAreaOUTIntermLabel;
            denAccum                 =  lab->d_denAccumIntermLabel;
            densityLag               =  lab->d_densityLagInterm_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCIntermLabel;
            flowIN                   =  lab->d_totalflowINIntermLabel;
            flowOUT                  =  lab->d_totalflowOUTIntermLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolIntermLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorIntermLabel;
            integrator_last_step     =  false;
            integrator_step_name     =  "BEEmulation2";
            integrator_step_number   =  TimeIntegratorStepNumber::Second;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuessInterm_label;
            pressure_guess           =  lab->d_pressurePredLabel;
            pressure_out             =  lab->d_pressureIntermLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityInterm_label;
            ref_pressure             =  lab->d_refPressureInterm_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolIntermLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorIntermLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyIntermLabel;
            ummsExactSol             =  lab->d_totalummsExactSolIntermLabel;
            ummsLnError              =  lab->d_totalummsLnErrorIntermLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolIntermLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorIntermLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolIntermLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorIntermLabel;
          break;

          case TimeIntegratorStepType::BEEmulation3:
            areaOUT                  =  lab->d_totalAreaOUTLabel;
            denAccum                 =  lab->d_denAccumLabel;
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            floutbc                  =  lab->d_netflowOUTBCLabel;
            flowIN                   =  lab->d_totalflowINLabel;
            flowOUT                  =  lab->d_totalflowOUTLabel;
            gradpmmsExactSol         =  lab->d_totalgradpmmsExactSolLabel;
            gradpmmsLnError          =  lab->d_totalgradpmmsLnErrorLabel;
            integrator_last_step     =  true;
            integrator_step_name     =  "BEEmulation3";
            integrator_step_number   =  TimeIntegratorStepNumber::Third;
            multiple_steps           =  true;
            negativeDensityGuess     =  lab->d_negativeDensityGuess_label;
            pressure_guess           =  lab->d_pressureIntermLabel;
            pressure_out             =  lab->d_pressurePSLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensity_label;
            ref_pressure             =  lab->d_refPressure_label;
            smmsExactSol             =  lab->d_totalsmmsExactSolLabel;
            smmsLnError              =  lab->d_totalsmmsLnErrorLabel;
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            tke_out                  =  lab->d_totalKineticEnergyLabel;
            ummsExactSol             =  lab->d_totalummsExactSolLabel;
            ummsLnError              =  lab->d_totalummsLnErrorLabel;
            use_old_values           =  true;
            vmmsExactSol             =  lab->d_totalvmmsExactSolLabel;
            vmmsLnError              =  lab->d_totalvmmsLnErrorLabel;
            wmmsExactSol             =  lab->d_totalwmmsExactSolLabel;
            wmmsLnError              =  lab->d_totalwmmsLnErrorLabel;
          break;

          default: 
            throw InvalidValue("Unknown explicit time integrator type", __FILE__, __LINE__);
        }
        
        catchUninitializedVars();
      }; 

    ~TimeIntegratorLabel() {};

    private:
    TimeIntegratorLabel(const TimeIntegratorLabel&);
    TimeIntegratorLabel& operator=(const TimeIntegratorLabel&);
  };
} // End namespace Uintah

#endif
