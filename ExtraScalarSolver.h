//----- ExtraScalarSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExtraScalarSolver_h
#define Uintah_Component_Arches_ExtraScalarSolver_h

/**************************************
CLASS
   ExtraScalarSolver
   
   Class ExtraScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ExtraScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ExtraScalarSolver linearizes and solves scalar
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesConstVariables.h>

namespace Uintah {
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class TurbulenceModel;
  class PhysicalConstants;
  class Discretization;
  class Source;
  class BoundaryCondition;
  class RHSSolver;
  class TimeIntegratorLabel;
  class ExtraScalarSrc;

class ExtraScalarSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of the Scalar solver.
      // PRECONDITIONS
      // POSTCONDITIONS
      //   A linear level solver is partially constructed.  
      ExtraScalarSolver(const ArchesLabel* label, const MPMArchesLabel* MAlb, 
		   PhysicalConstants* physConst);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~ExtraScalarSolver();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      void problemSetup(const ProblemSpecP& params);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Schedule Solve of linearized scalar equation
      void solve(SchedulerP& sched,
		 const PatchSet* patches,
		 const MaterialSet* matls,
		 const TimeIntegratorLabel* timelabels,
                 bool d_EKTCorrection,
                 bool doing_EKT_now);
   
      ///////////////////////////////////////////////////////////////////////
      // Schedule Build of linearized matrix
      void sched_buildLinearMatrix(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels,
                                   bool d_EKTCorrection,
                                   bool doing_EKT_now);

      ///////////////////////////////////////////////////////////////////////
      // Schedule Linear Solve for Scalar
      void sched_scalarLinearSolve(SchedulerP&, const PatchSet* patches,
				   const MaterialSet* matls,
				   const TimeIntegratorLabel* timelabels,
                                   bool d_EKTCorrection,
                                   bool doing_EKT_now);

      inline void setMMS(bool doMMS) {
        d_doMMS=doMMS;
      }
      inline const VarLabel* getScalarLabel() {
        return d_scalar_label;
      }
      inline const double getScalarInitValue() {
        return d_scalar_init_value;
      }
      inline const string getScalarName() {
        return d_scalar_name;
      }
      inline void setTurbulenceModel(TurbulenceModel* turbModel) {
        d_turbModel = turbModel;
      }
      inline void setBoundaryCondition(BoundaryCondition* boundaryCondition) {
        d_boundaryCondition = boundaryCondition;
      }
      inline const bool isDensityWeighted() {
        return d_scalar_density_weighted;
      }
      inline const bool useforDen(){
	//if true, will use this scalar to compute rho
	return d_scalar_useforden;
	//warning!  This assumes (for now) that only one scalar
	//           can be used from all Extrascalars.  
      }
protected:

private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the solver.
      ExtraScalarSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      // Actually Build the linear matrix
      //    [in] 
      //        add documentation here
      void buildLinearMatrix(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels,
                             bool d_EKTCorrection,
                             bool doing_EKT_now);

      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Scalar
      //    [in] 
      //        add documentation here
      void scalarLinearSolve(const ProcessorGroup* pc,
			     const PatchSubset* patches,
			     const MaterialSubset* /*matls*/,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw,
			     const TimeIntegratorLabel* timelabels,
                             bool d_EKTCorrection,
                             bool doing_EKT_now);


private:
      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      ArchesVariables* d_scalarVars;
      // computes coefficients
      Discretization* d_discretize;
      // computes sources
      Source* d_source;
      // linear solver
      RHSSolver* d_rhsSolver;
      // turbulence model
      TurbulenceModel* d_turbModel;
      // boundary condition
      BoundaryCondition* d_boundaryCondition;
      // physical constants
      PhysicalConstants* d_physicalConsts;
      int d_conv_scheme;
      int d_limiter_type;
      int d_boundary_limiter_type;
      bool d_central_limiter;
      bool d_dynScalarModel;
      double d_turbPrNo;
      bool d_doMMS;
#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif
      string d_scalar_name;
      double d_scalar_init_value;
      bool d_scalar_diffusion;
      bool d_scalar_density_weighted;
      bool d_scalar_useforden;
      const VarLabel* d_scalar_label;
      const VarLabel* d_scalar_temp_label;
      const VarLabel* d_scalar_coef_label;
      const VarLabel* d_scalar_diff_coef_label;
      const VarLabel* d_scalar_nonlin_src_label;

      bool d_calcExtraScalarSrcs;
      ExtraScalarSrc* d_extraScalarSrc;
      vector<ExtraScalarSrc*> d_extraScalarSources;


}; // End class ExtraScalarSolver
} // End namespace Uintah


#endif

