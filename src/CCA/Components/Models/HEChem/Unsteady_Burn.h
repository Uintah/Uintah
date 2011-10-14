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



#ifndef Packages_Uintah_CCA_Components_Examples_Unsteady_Burn_h
#define Packages_Uintah_CCA_Components_Examples_Unsteady_Burn_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
/**************************************

CLASS
   Unsteady_Burn
  

GENERAL INFORMATION

   Unsteady_Burn.h

   Changwei Xiong
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Unsteady_Burn

DESCRIPTION
   This is one of the general class of reaction models that converts solid 
   material into gas (combustion of high energy material).  The fomalism is
   based on the Ward, Son, Brewster (WSB) algebraic model of HE combustion.
   At each time step, the unsteady burn rate (mass transfer rate) of the solid 
   material is computed by calculating the updated burning surface temperature
   and updated burning surface temperature gradient.
  
WARNING
  
****************************************/

  class Unsteady_Burn : public ModelInterface {
  public:
    Unsteady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params,
                  const ProblemSpecP& prob_spec);
    virtual ~Unsteady_Burn();

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
    void computeModelSources(const ProcessorGroup*, const PatchSubset*,
                             const MaterialSubset*, DataWarehouse*, 
                             DataWarehouse*, const ModelInfo*);

    double computeBurnedMass(double To, double P, double Vc, double surfArea, double delT,
                             double solidMass, double& beta, double& Ts, Vector& dx);
    
    Unsteady_Burn(const Unsteady_Burn&);
    Unsteady_Burn& operator=(const Unsteady_Burn&);

    const VarLabel* BurningCellLabel;
    const VarLabel* TsLabel;
    const VarLabel* BetaLabel;
    const VarLabel* PartBetaLabel;
    const VarLabel* PartTsLabel;
    const VarLabel* totalMassBurnedLabel;
    const VarLabel* totalHeatReleasedLabel;
    
    ProblemSpecP d_params;
    ProblemSpecP d_prob_spec;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   
    
    MPMICELabel* MIlb;
    ICELabel* Ilb;
    MPMLabel* Mlb;
    MaterialSet* mymatls;

    // flags for the conservation test
    struct saveConservedVars{
      bool onOff;
      bool mass;
      bool energy;
      };
    saveConservedVars* d_saveConservedVars;
    
    double R ;   /* IdealGasConst      */
    double Ac;   /* PreExpCondPh          */
    double Ec;   /* ActEnergyCondPh in unit of Temperature */
    double Bg;   /* PreExpGasPh           */
    double Qc;   /* CondPhaseHeat         */
    double Qg;   /* GasPhaseHeat          */
    double Kg;   /* HeatConductGasPh      */
    double Kc;   /* HeatConductCondPh     */
    double Cp;   /* SpecificHeatCondPh    */
    double MW;   /* MoleWeightGasPh       */
    double BP;   /* Number of Particles at Boundary */
    double Bm;   /* BurnrateModCoef       */
    double Nc;   /* CondPhaseUnsteadyCoef */
    double Ng;   /* GasPhaseUnsteadyCoef  */      
    double ThresholdPressure; /*Threshold Press for burning */
    double ignitionTemp;      /* IgnitionTemp               */

    double MIN_MASS_IN_A_CELL;
      
    double CC1;  /* CC1 = Ac*R*Kc*Ec/Cp        */
    double CC2;  /* CC2 = Qc/Cp/2              */
    double CC3;  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    double CC4;  /* CC4 = Qc/Cp                */
    double CC5;  /* CC5 = Qg/Cp                */
    
    // Structure used to pass values into the iterator.  This is used to 
    //   prevent global definitions that may be corrupted by 
    //   multiple versions working on the same variables.
    typedef struct {
      /* C's, L's & R's, Tmin & Tmax, T_ignition are updated in UpdateConstants function  */
      double C1;   /* C1 = CC1 / Vc, (Vc = Condensed Phase Specific Volume) */
      double C2;   /* C2 = To + CC2      */
      double C3;   /* C3 = CC3 * P * P   */
      double C4;   /* C4 = To + CC4      */
      double C5;   /* C5 = CC5 * C3      */
        
      double NUM1;   /* K1 = Cp/Kc;                  */ 
      double NUM2;   /* K2 = Kc*rhoc/Cp;             */
      double NUM3;   /* K3 = -2*Kc/Qc;               */
      double NUM4;   /* K4 = 2*Ac*R*Kc*rhoc/(Ec*Qc); */
        
      double Tmin, Tmax; /* define the range of Ts */
      double IL, IR;     /* for interval update, left values and right values */
    } IterationVariable;
    
    void UpdateConstants(double To, double P, double Vc, IterationVariable *);
    double F_Ts(double Ts, IterationVariable *); /* function Ts = Ts(m(Ts))    */                    
    double Ts_m(double m, IterationVariable *); /* function Ts = Ts(m)    */
    double m_Ts(double Ts, IterationVariable *); /* function  m = m(Ts)    */

    double Func(double Ts, IterationVariable *);  /* function f = Ts - F_Ts(Ts) */
    double Deri(double Ts, IterationVariable *);  /* derivative of Func dF_dTs  */
    
    double Ts_max(IterationVariable *);
    void SetInterval(double f, double Ts, IterationVariable *);
    double BisectionNewton(double Ts, IterationVariable *);
    
    static const double EPSILON;   /* stop epsilon for Bisection-Newton method */
    static const double INIT_TS;   /* initial surface temperature          */
    static const double INIT_BETA; /* initial surface temperature gradient */
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO  1e-12
  };
}

#endif










