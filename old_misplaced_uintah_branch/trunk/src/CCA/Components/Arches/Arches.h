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
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

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
	 
      ///////////////////////////////////////////////////////////////////////
      // Schedule parameter initialization
      virtual void sched_paramInit(const LevelP& level,
				   SchedulerP&);
      
      ///////////////////////////////////////////////////////////////////////
      // Schedule Compute if Stable time step
      virtual void scheduleComputeStableTimestep(const LevelP& level,
						 SchedulerP&);

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
      // GROUP: Access Functions :
      ///////////////////////////////////////////////////////////////////////
	// Boolean to see whether or not Enthalpy is solved for
      inline bool checkSolveEnthalpy() const{
	return d_calcEnthalpy;
      }

      // for multimaterial
      void setMPMArchesLabel(const MPMArchesLabel* MAlb)
	{
	  d_MAlab = MAlb;
	}

      const ArchesLabel* getArchesLabel()
	{
	  return d_lab;
	}
      NonlinearSolver* getNonlinearSolver()
	{
	  return d_nlSolver;
	}

      BoundaryCondition* getBoundaryCondition()
	{
	  return d_boundaryCondition;
	}

      TurbulenceModel* getTurbulenceModel()
	{
	  return d_turbModel;
	}
      virtual double recomputeTimestep(double current_dt);
      
      virtual bool restartableTimesteps();


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

private:

      double d_deltaT;
      bool d_variableTimeStep;
      bool d_calcScalar;
      bool d_calcReactingScalar;
      bool d_calcEnthalpy;
      bool d_calcVariance;
      bool d_mixedModel;
      bool d_doMMS;
      bool d_extraProjection;
      bool d_EKTCorrection;
      string turbModel;
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
      // for multimaterial
      const MPMArchesLabel* d_MAlab;
#ifdef PetscFilter
      Filter* d_filter;
#endif

//      int nofTimeSteps;
#ifdef multimaterialform
      MultiMaterialInterface* d_mmInterface;
#endif

    string d_timeIntegratorType;

    bool d_recompile;

    AnalysisModule* d_analysisModule;
    
    bool d_set_initial_condition;
    std::string d_init_inputfile;
    TimeIntegratorLabel* init_timelabel;
    bool init_timelabel_allocated;
    bool d_dynScalarModel;
    bool d_underflow;
    string d_mms;
    //linear mms
    double cu,cv,cw,cp,phi0;
    //sine mms
    double amp;
    bool d_calcExtraScalars;
    ExtraScalarSolver* d_extraScalarSolver;
    vector<ExtraScalarSolver*> d_extraScalars;
    

}; // end class Arches

} // End namespace Uintah

#endif

