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

/**
 * \file
 *
 * \mainpage
 *
 * \author Various Generations of CRSim
 * \date   September 1997 (?)
 *
 * Arches is a variable density, low-mach (pressure projection) CFD formulation, with radiation, chemistry and
 * turbulence subgrid closure.
 *
 */

//----- Arches.h -----------------------------------------------

#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/**************************************

CLASS
   Arches

   Short description...

GENERAL INFORMATION

   Arches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 University of Utah

KEYWORDS
   Arches

DESCRIPTION
   Long description...

WARNING

****************************************/

#include <sci_defs/petsc_defs.h>
#include <sci_defs/uintah_defs.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <string>

// Divergence constraint instead of drhodt in pressure equation
//#define divergenceconstraint

// Exact Initialization for first time step in
// MPMArches problem, to eliminate problem of
// sudden appearance of mass in second step
//
// #define ExactMPMArchesInitialize


#ifdef HAVE_PETSC
  #define PetscFilter
#endif

// Filtering of drhodt is now an input parameter for Properties

# ifdef WASATCH_IN_ARCHES
  namespace Wasatch{
    class Wasatch;
  }
# endif // WASATCH_IN_ARCHES

namespace Uintah {

  class VarLabel;
  class PhysicalConstants;
  class NonlinearSolver;
  class Properties;
  class TurbulenceModel;
  class ScaleSimilarityModel;
  class CompLocalDynamicProcedure;
  class BoundaryCondition;
  class MPMArchesLabel;
  class ArchesLabel;
  class TimeIntegratorLabel;
  class ExtraScalarSolver;
  class ExplicitTimeInt;
  class PartVel;
  class DQMOM;

#ifdef PetscFilter
  class Filter;
#endif

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

  // GROUP: Static Variables:
  ////////////////////////////////////////////////////////////////////////
  // Number of dimensions in the problem
  static const int NDIM;
  // GROUP: Constants:
  ////////////////////////////////////////////////////////////////////////
  enum d_eqnType { PRESSURE, MOMENTUM, SCALAR };
  enum d_dirName { NODIR, XDIR, YDIR, ZDIR };
  enum d_stencilName { AP, AE, AW, AN, AS, AT, AB };
  enum d_numGhostCells {ZEROGHOSTCELLS , ONEGHOSTCELL, TWOGHOSTCELLS,
                        THREEGHOSTCELLS, FOURGHOSTCELLS, FIVEGHOSTCELLS };
  int nofTimeSteps;
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Arches constructor
  Arches(const ProcessorGroup* myworld);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor
  virtual ~Arches();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  virtual void problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& materials_ps,
                            GridP& grid, SimulationStateP&);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule initialization
  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  virtual void restartInitialize();

  ///////////////////////////////////////////////////////////////////////
  // Schedule parameter initialization
  virtual void sched_paramInit(const LevelP& level,
                               SchedulerP&);

  ///////////////////////////////////////////////////////////////////////
  // Schedule Compute if Stable time step
  virtual void scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP&);
  void 
  MPMArchesIntrusionSetupForResart( const LevelP& level, SchedulerP& sched, 
      bool& recompile, bool doing_restart );

  ///////////////////////////////////////////////////////////////////////
  // Schedule time advance
  virtual void scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP&);

  ///////////////////////////////////////////////////////////////////////
   // Function to return boolean for recompiling taskgraph

    virtual bool needRecompile(double time, double dt,
                        const GridP& grid);

  virtual void sched_readCCInitialCondition(const LevelP& level,
                                            SchedulerP&);
  virtual void sched_interpInitialConditionToStaggeredGrid(const LevelP& level,
                                                           SchedulerP&);
  virtual void sched_getCCVelocities(const LevelP& level,
                                     SchedulerP&);
  virtual void sched_mmsInitialCondition(const LevelP& level,
                                         SchedulerP&);
  virtual void sched_weightInit( const LevelP& level,
                                SchedulerP& );
  virtual void sched_weightedAbsInit( const LevelP& level,
                                SchedulerP& );
  virtual void sched_scalarInit( const LevelP& level,
                                 SchedulerP& sched );

  // GROUP: Access Functions :
  ///////////////////////////////////////////////////////////////////////
    // Boolean to see whether or not Enthalpy is solved for
  inline bool checkSolveEnthalpy() const{
    return d_calcEnthalpy;
  }

  // for multimaterial
  void setMPMArchesLabel(const MPMArchesLabel* MAlb){
    d_MAlab = MAlb;
  }

  const ArchesLabel* getArchesLabel(){
    return d_lab;
  }
  NonlinearSolver* getNonlinearSolver(){
    return d_nlSolver;
  }

  BoundaryCondition* getBoundaryCondition(){
    return d_boundaryCondition;
  }

  TurbulenceModel* getTurbulenceModel(){
    return d_turbModel;
  }
  virtual double recomputeTimestep(double current_dt);

  virtual bool restartableTimesteps();

  void setWithMPMARCHES() {
    d_with_mpmarches = true;
  };


protected:

private:

  // GROUP: Constructors (Private):
  ////////////////////////////////////////////////////////////////////////
  // Default Arches constructor
  Arches();

  ////////////////////////////////////////////////////////////////////////
  // Arches copy constructor
  Arches(const Arches&);

  // GROUP: Overloaded Operators (Private):
  ////////////////////////////////////////////////////////////////////////
  // Arches assignment constructor
  Arches& operator=(const Arches&);

  // GROUP: Action Methods (Private):
  ////////////////////////////////////////////////////////////////////////
  // Arches assignment constructor
  void paramInit(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset*,
                 DataWarehouse* ,
                 DataWarehouse* new_dw );

  void computeStableTimeStep(const ProcessorGroup* ,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw);

  void readCCInitialCondition(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* ,
                              DataWarehouse* new_dw);

  void interpInitialConditionToStaggeredGrid(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset*,
                                             DataWarehouse* ,
                                             DataWarehouse* new_dw);

  void getCCVelocities(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset*,
                       DataWarehouse* ,
                       DataWarehouse* new_dw);

  void mmsInitialCondition(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* ,
                           DataWarehouse* new_dw);

  void weightInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);
  void weightedAbsInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

  void scalarInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );



private:

  /** @brief Registers all possible user defined source terms by instantiating a builder in the factory */
  void registerUDSources(ProblemSpecP& db);

  /** @brief Registers developer specific sources that may not have an input file specification */
  void registerSources();

  /** @brief Registers all possible models for DQMOM */
  void registerModels( ProblemSpecP& db );

  /** @brief Registers all possible equations by instantiating a builder in the factory */
  void registerTransportEqns(ProblemSpecP& db);

  /** @brief Registers all possible DQMOM equations by instantiating a builder in the factory */
  void registerDQMOMEqns(ProblemSpecP& db);

  /** @brief Registers all possible Property Models by instantiating a builder in the factory */
  void registerPropertyModels( ProblemSpecP& db );

# ifdef WASATCH_IN_ARCHES
  Wasatch::Wasatch* const d_wasatch;
# endif // WASATCH_IN_ARCHES

      double d_init_dt; // The initial dt from input file.
      double d_init_mix_frac; // The initial value of mixture fraction in the domain (for paramInit)
      bool d_variableTimeStep;
      string d_whichTurbModel;
      bool d_calcScalar;
      bool d_calcReactingScalar;
      bool d_calcEnthalpy;
      bool d_calcNewEnthalpy;
      bool d_calcVariance;
      bool d_mixedModel;
      bool d_doMMS;
      bool d_with_mpmarches;
      bool d_extraProjection;

      ScaleSimilarityModel* d_scaleSimilarityModel;
      CompLocalDynamicProcedure* d_initTurb;
      PhysicalConstants* d_physicalConsts;
      NonlinearSolver* d_nlSolver;
      // properties...solves density, temperature and species concentrations
      Properties* d_props;
      // Turbulence Model
      TurbulenceModel* d_turbModel;
      // Boundary conditions
      BoundaryCondition* d_boundaryCondition;
      SimulationStateP d_sharedState;
      // Variable labels that are used by the simulation controller
      ArchesLabel* d_lab;

      const MPMArchesLabel* d_MAlab;
#ifdef PetscFilter
      Filter* d_filter;
#endif

//      int nofTimeSteps;

    string d_timeIntegratorType;

    AnalysisModule* d_analysisModule;

    bool d_set_initial_condition;
    std::string d_init_inputfile;
    TimeIntegratorLabel* init_timelabel;
    bool init_timelabel_allocated;
    bool d_dynScalarModel;
    bool d_underflow;
    string d_mms;
    //linear mms
    double d_cu,d_cv,d_cw,d_cp,d_phi0,d_esphi0;
    //sine mms
    double d_amp;

    // Variables----
    vector<string> d_scalarEqnNames;
    bool d_doDQMOM; // do we need this as a private member?
    std::string d_which_dqmom;
    int d_tOrder;
    ExplicitTimeInt* d_timeIntegrator;
    PartVel* d_partVel;
    DQMOM* d_dqmomSolver;

    bool d_doingRestart;
    bool d_newBC_on_Restart;


}; // end class Arches

} // End namespace Uintah

#endif

