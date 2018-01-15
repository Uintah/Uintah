/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
    std::string integrator_step_name;
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
    const VarLabel* negativeDensityGuess;
    const VarLabel* densityLag;

    //__________________________________
    //  initialize all the variabless
    void initializeVars(){
      densityLag               =  nullptr;
      factor_divide            = -9;                 
      factor_new               = -9;                 
      factor_old               = -9;                 
      integrator_step_name     = "-9";
      integrator_step_number   =  -9;
      negativeDensityGuess     =  nullptr;
      pressure_guess           =  nullptr;
      pressure_out             =  nullptr;
      ref_density              =  nullptr;
      time_multiplier          = -9;
      time_position_multiplier_after_average    = -9;
      time_position_multiplier_before_average   = -9;
    }
    
    //__________________________________
    //  Test for uninitialized Variables
    void catchUninitializedVars(){
      bool test =( ( densityLag               ==  nullptr ) ||
      ( factor_divide            == -9 )    ||              
      ( factor_new               == -9 )    ||              
      ( factor_old               == -9 )    ||              
      ( integrator_step_name     == "-9" )  ||
      ( integrator_step_number   ==  -9 )   ||
      ( negativeDensityGuess     ==  nullptr ) ||
      ( pressure_guess           ==  nullptr ) ||
      ( pressure_out             ==  nullptr ) ||
      ( ref_density              ==  nullptr ) ||
      ( time_multiplier          == -9 )    ||
      ( time_position_multiplier_after_average    == -9 ) ||
      ( time_position_multiplier_before_average   == -9 ) );
      
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
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            time_multiplier          =  1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::FE:
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::Predictor:
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::OldPredictor:
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
            integrator_last_step     =  false;
            integrator_step_name     =  "OldPredictor";
            integrator_step_number   =  TimeIntegratorStepNumber::First;
            multiple_steps           =  true;
            pressure_guess           =  lab->d_pressurePSLabel;
            pressure_out             =  lab->d_pressurePredLabel;
            recursion                =  false;
            ref_density              =  lab->d_refDensityPred_label;
            ref_pressure             =  lab->d_refPressurePred_label;
            time_multiplier          =  0.5;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::Corrector:
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  2.0;
            factor_new               =  1.0;
            factor_old               =  1.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 2.0;
            use_old_values           =  false;
          break;

          case TimeIntegratorStepType::OldCorrector:
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::CorrectorRK3:
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  3.0;
            factor_new               =  2.0;
            factor_old               =  1.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.5;
            use_old_values           =  false;
          break;

          case TimeIntegratorStepType::Intermediate:
            densityLag               =  lab->d_densityLagInterm_label;
            factor_divide            =  4.0;
            factor_new               =  1.0;
            factor_old               =  3.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 0.5;
            time_position_multiplier_before_average = 2.0;
            use_old_values           =  false;
          break;

          case TimeIntegratorStepType::BEEmulation1:
            densityLag               =  lab->d_densityLagPred_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::BEEmulation2:
            densityLag               =  lab->d_densityLagInterm_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
          break;

          case TimeIntegratorStepType::BEEmulation3:
            densityLag               =  lab->d_densityLag_label;
            factor_divide            =  1.0;
            factor_new               =  1.0;
            factor_old               =  0.0;
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
            time_multiplier          =  1.0;
            time_position_multiplier_after_average = 1.0;
            time_position_multiplier_before_average = 1.0;
            use_old_values           =  true;
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
