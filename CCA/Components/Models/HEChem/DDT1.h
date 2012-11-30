/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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
			      ModelSetup* setup);

      
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

                                             
   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches,
                                         const ModelInfo* mi);


  private:    
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
                             double solidMass);
    
    double computeInductionAngle(IntVector *nodeIdx, 
                                  constNCVariable<double> &rctMass_NC, 
                                  constNCVariable<double> &NC_CCweight, 
                                  Vector &dx, 
                                  double& cos_theta, 
                                  double& theta,
                                  Point hotcellCord, 
                                  Point cellCord);  
      
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
    enum typeofBurning{ NOTDEFINED, WARMINGUP, CONDUCTIVE, CONVECTIVE, ONSURFACE };

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
   

    string fromMaterial, toMaterial, burnMaterial;
    // Detonation Model
    bool   d_useZeroOrderRate;
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
    double R ;  /* IdealGasConst      */
    double Ac;  /* PreExpCondPh       */
    double Ec;  /* ActEnergyCondPh in unit of Temperature    */
    double Bg;  /* PreExpGasPh        */
    double Qc;  /* CondPhaseHeat      */
    double Qg;  /* GasPhaseHeat       */
    double Kg;  /* HeatConductGasPh   */
    double Kc;  /* HeatConductCondPh  */
    double Cp;  /* SpecificHeatCondPh */
    double MW;  /* MoleWeightGasPh    */
    double BP;  /* Number of Particles at Boundary          */
    double d_thresholdPress_SB; /*Threshold Press for burning */
    double ignitionTemp;        /* IgnitionTemp  */
    
    //Induction Model
    bool   d_useInductionTime;
    double d_IC; /* Ignition time constant                           IgnitionConst */
    double d_Fb; /* Preexponential for suface flame spread equation  PreexpoConst */
    double d_Fc; /* exponent term for suface flame spread equaiton   ExponentialConst */
    double d_PS; /* P0 for surface flame spread equation             PressureShift */  
      
    double MIN_MASS_IN_A_CELL;
    
    double CC1; /* CC1 = Ac*R*Kc*Ec/Cp        */
    double CC2; /* CC2 = Qc/Cp/2              */
    double CC3; /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    double CC4; /* CC4 = Qc/Cp                */
    double CC5; /* CC5 = Qg/Cp                */
      
    /* C's, IL & IR, Tmin & Tmax are updated in UpdateConstants function  */
    // Structure used to pass values into the iterator.  This is used to 
    //   prevent global definitions that may be corrupted by 
    //   multiple versions working on the same variables.
    typedef struct {
      double C1; /* C1 = CC1 / Vc, (Vc = Condensed Phase Specific Volume) */
      double C2; /* C2 = To + CC2     */
      double C3; /* C3 = CC3 * P * P  */
      double C4; /* C4 = To + CC4     */
      double C5; /* C5 = CC5 * C3     */
          
      double Tmin, Tmax; /* define the range of Ts */
      double IL, IR;     /* for interval update, left values and right values */
    } IterationVariables;
      
    void UpdateConstants(double To, double P, double Vc, IterationVariables *iter);
    double F_Ts(double Ts, IterationVariables *iter); /* function Ts = Ts(m(Ts))    */                    
    double Ts_m(double m, IterationVariables *iter); /* function Ts = Ts(m)    */
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
      
      
    static const double EPSILON;   /* stop epsilon for Bisection-Newton method */   
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif
