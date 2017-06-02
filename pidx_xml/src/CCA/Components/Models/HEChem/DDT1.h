/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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



#ifndef Packages_Uintah_CCA_Components_Models_DDT1_h
#define Packages_Uintah_CCA_Components_Models_DDT1_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>

namespace Uintah {
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
/**************************************

CLASS
   DDT1
  

GENERAL INFORMATION

   DDT1.h

   Joseph Peterson
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   "DDT1" "JWL++" "Reactive Flow Model" "WSB"

DESCRIPTION
   Model for deflagration to detonation transition using WSB model for
   equation for deflagration and the detonation model based on "JWL++:  A Simple
   Reactive Flow Code Package for Detonation", P.Clark Souers, Steve Anderson,
   James Mercer, Estella McGuire and Peter Vitello, Propellants,
   Explosives, Pyrotechnics, 25, 54-58, 2000.
  
WARNING

****************************************/

  class DDT1 : public ModelInterface {
  public:
    DDT1(const ProcessorGroup* myworld, ProblemSpecP& params,
                const ProblemSpecP& prob_spec);

    virtual ~DDT1();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid, SimulationStateP& sharedState,
			      ModelSetup* setup, const bool isRestart);

      
    virtual void scheduleInitialize(SchedulerP&,
				    const LevelP& level,
				    const ModelInfo*);

    virtual void initialize(const ProcessorGroup*,
                            const PatchSubset*,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse*);

    virtual void restartInitialize() {}
      
    virtual void scheduleComputeStableTimestep(SchedulerP&,
					       const LevelP& level,
					       const ModelInfo*);
      
 
    virtual void scheduleComputeModelSources(SchedulerP&,
						   const LevelP& level,
						   const ModelInfo*);
                                             
    virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                               const LevelP&,
                                               const MaterialSet*);
                                               
   virtual void computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,
                                    DataWarehouse*,
                                    const int);
                                    
   virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched);
                                      
   virtual void scheduleRefine( const PatchSet* patches,
                                SchedulerP& sched );
                                             
   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches,
                                         const ModelInfo* mi);


  private:    
  
    bool isDoubleEqual(double a, double b);
    
    void problemSetup_BulletProofing(ProblemSpecP& ps);
    
    
    void computeBurnLogic(const ProcessorGroup*, 
                          const PatchSubset*,
                          const MaterialSubset*, 
                          DataWarehouse*, 
                          DataWarehouse*, 
                          const ModelInfo*);
                             
    void computeModelSources(const ProcessorGroup*, 
                             const PatchSubset*,
                             const MaterialSubset*, 
                             DataWarehouse*, 
                             DataWarehouse*, 
                             const ModelInfo*);
      
    void computeNumPPC(const ProcessorGroup*, 
                       const PatchSubset*,
                       const MaterialSubset*, 
                       DataWarehouse*, 
                       DataWarehouse*, 
                       const ModelInfo*);
      
    double computeBurnedMass(double To, 
                             double& Ts,  
                             double P, 
                             double Vc,
                             double surfArea, 
                             double delT, 
                             double solidMass,
                             const double min_mass_in_a_cell);
    
    double computeInductionAngle( IntVector *nodeIdx, 
                                  constNCVariable<double> &rctMass_NC, 
                                  constNCVariable<double> &NC_CCweight, 
                                  Vector &dx, 
                                  double& cos_theta, 
                                  double& theta,
                                  Point hotcellCord, 
                                  Point cellCord);

    void refine( const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset* /*matls*/,
                 DataWarehouse* ,
                 DataWarehouse* new_dw );
      
    DDT1(const DDT1&);
    DDT1& operator=(const DDT1&);

      
    // Simple Burn
    const VarLabel* detLocalToLabel;        // diagnostic labels
    const VarLabel* onSurfaceLabel;         // diagnostic labels
    const VarLabel* surfaceTempLabel;
    const VarLabel* totalMassBurnedLabel;
    const VarLabel* totalHeatReleasedLabel;
    const VarLabel* burningLabel;   
    const VarLabel* crackedEnoughLabel;   
    const VarLabel* TsLabel;
    const VarLabel* numPPCLabel;
    const VarLabel* burningTypeLabel;
    
    // JWL++
    const VarLabel* reactedFractionLabel;   // diagnostic labels
    const VarLabel* delFLabel;              // diagnostic labels
    const VarLabel* totalMassConvertedLabel;
    const VarLabel* detonatingLabel;
    const VarLabel* inductionTimeLabel;
    const VarLabel* countTimeLabel;
    const VarLabel* BurningCriteriaLabel;
    const VarLabel* adjOutIntervalsLabel;
    
    enum typeofBurning{ NOTDEFINED, WARMINGUP, CONDUCTIVE, CONVECTIVE, ONSURFACE };
    enum {ZERO, PRESSURE_EXCEEDED, DETONATION_DETECTED};

    const VarLabel* pCrackRadiusLabel;
    
    ProblemSpecP d_prob_spec;
    ProblemSpecP d_params;
    const Material* d_matl0;
    const Material* d_matl1;
    const Material* d_matl2;
    SimulationStateP d_sharedState;   

    ICELabel* Ilb;
    MPMICELabel* MIlb;
    MPMLabel* Mlb;
    MaterialSet* d_mymatls;
    MaterialSubset* d_one_matl;
   

    std::string fromMaterial, toMaterial, burnMaterial;
    // Detonation Model
    double d_G;
    double d_b;
    double d_E0;
    double d_threshold_press_JWL;    // JWL++
    double d_threshold_volFrac;

    // Cracking Model
    bool   d_useCrackModel;
    double d_Gcrack;            // Crack Burning Rate Constant
    double d_nCrack;            // Crack Burning Pressure Exponent
    double d_crackVolThreshold; // for cracking

    // Burn Model
    double d_R ;  /* IdealGasConst      */
    double d_Ac;  /* PreExpCondPh       */
    double d_Ec;  /* ActEnergyCondPh in unit of Temperature    */
    double d_Bg;  /* PreExpGasPh        */
    double d_Qc;  /* CondPhaseHeat      */
    double d_Qg;  /* GasPhaseHeat       */
    double d_Kg;  /* HeatConductGasPh   */
    double d_Kc;  /* HeatConductCondPh  */
    double d_Cp;  /* SpecificHeatCondPh */
    double d_MW;  /* MoleWeightGasPh    */
    double d_BP;  /* Number of Particles at Boundary          */
    double d_thresholdPress_SB; /*Threshold Press for burning */
    double d_ignitionTemp;        /* IgnitionTemp  */
    
    //Induction Model
    bool   d_useInductionTime;
    double d_IC; /* Ignition time constant                           IgnitionConst */
    double d_Fb; /* Preexponential for suface flame spread equation  PreexpoConst */
    double d_Fc; /* exponent term for suface flame spread equaiton   ExponentialConst */
    double d_PS; /* P0 for surface flame spread equation             PressureShift */
    
    double d_CC1; /* d_CC1 = Ac*R*Kc*Ec/Cp        */
    double d_CC2; /* d_CC2 = Qc/Cp/2              */
    double d_CC3; /* d_CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    double d_CC4; /* d_CC4 = Qc/Cp                */
    double d_CC5; /* d_CC5 = Qg/Cp                */
      
    /* C's, IL & IR, Tmin & Tmax are updated in UpdateConstants function  */
    // Structure used to pass values into the iterator.  This is used to 
    //   prevent global definitions that may be corrupted by 
    //   multiple versions working on the same variables.
    typedef struct {
      double C1; /* C1 = d_CC1 / Vc, (Vc = Condensed Phase Specific Volume) */
      double C2; /* C2 = To + d_CC2     */
      double C3; /* C3 = d_CC3 * P * P  */
      double C4; /* C4 = To + d_CC4     */
      double C5; /* C5 = d_CC5 * C3     */
          
      double Tmin, Tmax; /* define the range of Ts */
      double IL, IR;     /* for interval update, left values and right values */
    } IterationVariables;
      
    void UpdateConstants(double To, double P, double Vc, IterationVariables *iter);
    double F_Ts(double Ts, IterationVariables *iter); /* function Ts = Ts(m(Ts))    */                    
    double Ts_m(double m, IterationVariables *iter);  /* function Ts = Ts(m)    */
    double m_Ts(double Ts, IterationVariables *iter); /* function  m = m(Ts)    */
      
    double Func(double Ts, IterationVariables *iter);  /* function f = Ts - F_Ts(Ts) */
    double Deri(double Ts, IterationVariables *iter);  /* derivative of Func dF_dTs  */
      
    double Ts_max(IterationVariables *iter);
    void SetInterval(double f, double Ts, IterationVariables *iter);
    double BisectionNewton(double Ts, IterationVariables *iter);
      
    bool d_is_mpm_matl;  // Is matl 0 a mpm_matl?
    double d_cv_0;      //specific heat
    
    // flags for the conservation test
    struct saveConservedVars{
        bool onOff;
        bool mass;
        bool energy;
    };
    saveConservedVars* d_saveConservedVars;
      
    
    //__________________________________
    // struct used to adjust the I/O intervals based 
    // on either a pressure threshold exceeded or a detonation has been detected
    struct adj_IO{                  // pressure_switch
      bool onOff;                   // is this option on or off?
      double timestepsLeft;         // timesteps left until sus shuts down
      double pressThreshold;
      double output_interval;       // output interval in physical seconds
      double chkPt_interval;        // checkpoing interval in physical seconds
      ~adj_IO(){};
    };  
    adj_IO* d_adj_IO_Press;
    adj_IO* d_adj_IO_Det;  
      
    static const double d_EPSILON;   /* stop epsilon for Bisection-Newton method */   
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
