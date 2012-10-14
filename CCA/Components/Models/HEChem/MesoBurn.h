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



#ifndef Packages_Uintah_CCA_Components_Examples_Meso_Burn_h
#define Packages_Uintah_CCA_Components_Examples_Meso_Burn_h

#include <CCA/Ports/ModelInterface.h>
#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {
  class ICELabel;
  class MPMLabel;
  class MPMICELabel;
/**************************************

CLASS
   MesoBurn
  

GENERAL INFORMATION

   MesoBurn.h

   Joseph R. Peterson
   Department of Chemistry
   University of Illinois at Urbana-Champaign

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   MesoBurn

DESCRIPTION
   This is one of the general class of reaction models that converts solid 
   material into gas (combustion of high energy material).  The fomalism is
   based on the Ward, Son, Brewster (WSB) algebraic model of HE combustion.
   At each time step, the burn rate (mass transfer rate) of the solid 
   material is computed by calculating the surface temperature expected for
   a steady burn (based on the bulk solid temperature, gas pressure or 
   density, thermal conductivity and heat capacity of the solid.  The main
   assumption is that the temperature distributions in both the solid and
   gas are at equilibrium at each time step in the simulation.  
 
   This is a modified version of Steady_Burn model with a modified ignition
   criterion.  It is based on an adiabatic induction time from:
 
   Menikoff, R. Detonation Waves in PBX 9501, Combustion Theory and Modelling,
     10, pp 1003-1021 (2006).
 
   of the form:
  
   tadb=[T^2*Cv/TaQ]exp(Ta/T)/k
 
WARNING
  
****************************************/

  class MesoBurn : public ModelInterface {
  public:
    MesoBurn(const ProcessorGroup* myworld, ProblemSpecP& params,
                const ProblemSpecP& prob_spec);
    virtual ~MesoBurn();

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
    
    void computeParticleVariables(const ProcessorGroup*, const PatchSubset*,
                                  const MaterialSubset*, DataWarehouse*, 
                                  DataWarehouse*, const ModelInfo*);
    
    double computeBurnedMass(double To, double& Ts,  double P, double Vc,
                             double surfArea, double delT, double solidMass);
    
    MesoBurn(const MesoBurn&);
    MesoBurn& operator=(const MesoBurn&);
    
    // Diagnostic Variables
    const VarLabel* BurningCellLabel;
    const VarLabel* TsLabel;
    const VarLabel* numPPCLabel;
    const VarLabel* inductionTimeLabel;      // smallest induction time of any particle in cell
    const VarLabel* inductionTimePartLabel;  // the 1/100 weighted induction time 
    const VarLabel* inducedLabel;       // 0-1, % of particles reacting
    const VarLabel* inducedMassLabel;   // amount of mass in cell that is reacting
    const VarLabel* timeInducedLabel; // Amount of time a particular particle has been inducint
      
    // Cummulative Variables
    const VarLabel* totalMassBurnedLabel;
    const VarLabel* totalHeatReleasedLabel;
    const VarLabel* totalSurfaceAreaLabel;
    
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
      bool surfaceArea;
    };
    saveConservedVars* d_saveConservedVars;
    
    // Combustion Variables
    double R ;  /* IdealGasConst      (J/molK)   */
    double Ac;  /* PreExpCondPh       (s^-1)     */
    double Ec;  /* ActEnergyCondPh    (J/mol)    */
    double Bg;  /* PreExpGasPh        (m^3/skgK) */
    double Qc;  /* CondPhaseHeat      (J/kg)     */
    double Qg;  /* GasPhaseHeat       (J/kg)     */
    double Kg;  /* HeatConductGasPh   (W/mK)     */
    double Kc;  /* HeatConductCondPh  (W/mK)     */
    double Cp;  /* SpecificHeatCondPh (J/kgK)    */
    double MW;  /* MoleWeightGasPh    (kg/mol)   */
    double BP;  /* Number of Particles at Boundary          */
    double ThresholdPressure; /*Threshold Press for burning (Pa) */
      
    // Ignition Variables
    double Ta;      // Activation Temperature, Ea/R (K) 
    double k;       // Rate constant (s^-1) 
    double Cv;      // Specific heat at constant volume (J/kgK)
    bool afterMelting; // True if the material must be melted 
                       //  before reaction can begin.  This requires
                       //  the constitutive model for the reactant
                       //  material have a melting temperature model.
    int resolution;    // Resolution of a ground state cell in particles
      
    double MIN_MASS_IN_A_CELL;
    
    double CC1; /* CC1 = Ac*R*Kc*Ec/Cp        */
    double CC2; /* CC2 = Qc/Cp/2              */
    double CC3; /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    double CC4; /* CC4 = Qc/Cp                */
    double CC5; /* CC5 = Qg/Cp                */
    
    /* C's, IL & IR, Tmin & Tmax are updated in UpdateConstants function  */
    // Structure used to pass values into the iterator.  This is used to 
    //   prevent global definitions that may be corrupted by 
    //   multiple versions in a threaded environment from working on 
    //   the same variables.
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
    
    static const double EPSILON;   /* stop epsilon for Bisection-Newton method */
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO  1e-12
  };
}

#endif // Packages_Uintah_CCA_Components_Examples_Meso_Burn_h
