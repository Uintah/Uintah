
#ifndef Packages_Uintah_CCA_Components_Examples_Unsteady_Burn_h
#define Packages_Uintah_CCA_Components_Examples_Unsteady_Burn_h

#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>

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
    Unsteady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params);
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

    virtual void setMPMLabel(MPMLabel* MLB);    
    
    
  private:    
    void computeModelSources(const ProcessorGroup*, const PatchSubset*,
			     const MaterialSubset*, DataWarehouse*, 
			     DataWarehouse*, const ModelInfo*);
    
    double computeSurfaceArea(Vector &rhoGradVector, Vector &dx);
    
    Vector computeDensityGradientVector(IntVector *nodeIdx, 
                                        constNCVariable<double> &NCsolidMass,
                                        constNCVariable<double> &NC_CCweight,
                                        Vector &dx);
    
    double computeBurnedMass(double To, double P, double Vc, double surfArea, double delT,
			     double solidMass, double& beta, double& Ts, Vector& dx);
    
    Unsteady_Burn(const Unsteady_Burn&);
    Unsteady_Burn& operator=(const Unsteady_Burn&);

    const VarLabel* BurningCellLabel;
    const VarLabel* TsLabel;
    const VarLabel* BetaLabel;
    const VarLabel* PartBetaLabel;
    const VarLabel* PartTsLabel;
    
    ProblemSpecP params;
    const Material* matl0;
    const Material* matl1;
    SimulationStateP d_sharedState;   
    
    MPMICELabel* MIlb;
    ICELabel* Ilb;
    MPMLabel* Mlb;
    MaterialSet* mymatls;
    
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
    
    double CC1;  /* CC1 = Ac*R*Kc*Ec/Cp        */
    double CC2;  /* CC2 = Qc/Cp/2              */
    double CC3;  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
    double CC4;  /* CC4 = Qc/Cp                */
    double CC5;  /* CC5 = Qg/Cp                */
    
    /* C's, L's & R's, Tmin & Tmax, T_ignition are updated in UpdateConstants function  */
    double C1;   /* C1 = CC1 / Vc, (Vc = Condensed Phase Specific Volume) */
    double C2;   /* C2 = To + CC2      */
    double C3;   /* C3 = CC3 * P * P   */
    double C4;   /* C4 = To + CC4      */
    double C5;   /* C5 = CC5 * C3      */

    double K1;   /* K1 = Cp/Kc;                  */ 
    double K2;   /* K2 = Kc*rhoc/Cp;             */
    double K3;   /* K3 = -2*Kc/Qc;               */
    double K4;   /* K4 = 2*Ac*R*Kc*rhoc/(Ec*Qc); */
    
    double T_ignition; /* T_ignition = C2        */
    double Tmin, Tmax; /* define the range of Ts */
    /* for interval update, left values and right values */
    double L0, R0;     
    double L1, R1;
    double L2, R2;
    double L3, R3; 
    
    void UpdateConstants(double To, double P, double Vc);
    double Fxn_Ts(double Ts); /* function Ts = f(Ts)    */
    double Fxn(double x);     /* function f = Ts -f(Ts) */
    double Ts_max();
    int Termination();        /* Convergence criteria   */
    double Secant(double u, double w);
    void SetInterval(double x);
    double Bisection(double l, double r);
    double BisectionSecant();

    static const double EPS;
    static const double UNDEFINED;
    static const double INIT_TS;
    static const double INIT_BETA;
    #define d_SMALL_NUM 1e-100
    #define d_TINY_RHO 1e-12
  };
}

#endif










