//----- PicardNonlinearSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_PicardNonlinearSolver_h
#define Uintah_Component_Arches_PicardNonlinearSolver_h

/**************************************
CLASS
   NonlinearSolver
   
   Class PicardNonlinearSolver is a subclass of NonlinearSolver
   which implements the Fixed Point Picard iteration.[Ref Kumar's thesis]

GENERAL INFORMATION
   PicardNonlinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000


KEYWORDS


DESCRIPTION
   Class PicardNonlinearSolver implements the
   Fixed Point Picard iteration method is used by
   ImplicitIntegrator to solve set of nonlinear equations

WARNING
   none
****************************************/

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/NonlinearSolver.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {
  using namespace SCIRun;
class PressureSolver;
class MomentumSolver;
class ScalarSolver;
class TurbulenceModel;
class Properties;
class BoundaryCondition;
class PhysicalConstants;
class EnthalpySolver; 
const double MACHINEPRECISSION = 14.0; //used to compute residual

class PicardNonlinearSolver: public NonlinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Solver initialized with all input data 
      PicardNonlinearSolver(const ArchesLabel* label,
			    const MPMArchesLabel* MAlb,
			    Properties* props, 
			    BoundaryCondition* bc,
			    TurbulenceModel* turbModel, 
			    PhysicalConstants* physConst,
			    bool calcEnthalpy,
			    const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for PicardNonlinearSolver.
      virtual ~PicardNonlinearSolver();


      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& input_db);

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Solve the nonlinear system. (also does some actual computations)
      // The code returns 0 if there are no errors and
      // 1 if there is a nonlinear failure.
      //    [in] 
      //        documentation here
      //    [out] 
      //        documentation here
      virtual int nonlinearSolve(const LevelP& level,
				 SchedulerP& sched);

  
      ///////////////////////////////////////////////////////////////////////
      // dummy Solve for proper initialization of MPMArches problem
      virtual int noSolve(const LevelP& level,
			  SchedulerP& sched);
  
      ///////////////////////////////////////////////////////////////////////
      // Schedule the Initialization of non linear solver
      //    [in] 
      //        data User data needed for solve 
      void sched_setInitialGuess(SchedulerP&, const PatchSet* patches,
				 const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the dummy solve
      void sched_dummySolve(SchedulerP& sched,
			    const PatchSet* patches,
			    const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the interpolation of velocities from Face Centered Variables
      //    to a Cell Centered Vector
      //    [in] 
      void sched_interpolateFromFCToCC(SchedulerP&, const PatchSet* patches,
				       const MaterialSet* matls,
				       const TimeIntegratorLabel* timelabels);

      void sched_probeData(SchedulerP&, const PatchSet* patches,
			   const MaterialSet* matls);

      // GROUP: Action Computations :
      ///////////////////////////////////////////////////////////////////////
      // Compute the residual
      //    [in] 
      //        documentation here
      double computeResidual(const LevelP&, 
			     SchedulerP& sched,
			     DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw);

protected :

private:

      // GROUP: Constructors (private):
      ////////////////////////////////////////////////////////////////////////
      // Should never be used
      PicardNonlinearSolver();

      // GROUP: Action Methods (private) :
      ///////////////////////////////////////////////////////////////////////
      // Actually Initialize the non linear solver
      //    [in] 
      //        data User data needed for solve 
      void setInitialGuess(const ProcessorGroup* pc,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Actual dummy solve
      void dummySolve(const ProcessorGroup* pc,
		      const PatchSubset* patches,
		      const MaterialSubset*,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Actually Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
      //    [in] 
      void interpolateFromFCToCC(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw,
				 const TimeIntegratorLabel* timelabels);

      void probeData(const ProcessorGroup* pc,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);

private:

      // const VarLabel*
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      // generation variable for DataWarehouse creation
  
      // Total number of nonlinear iterates
      int d_nonlinear_its;
      // for probing data for debuging or plotting
      bool d_probe_data;
      vector<IntVector> d_probePoints;
      // nonlinear residual tolerance
      double d_resTol;

      // properties...solves density, temperature and specie concentrations
      Properties* d_props;
      // Boundary conditions
      BoundaryCondition* d_boundaryCondition;
      // Turbulence Model
      TurbulenceModel* d_turbModel;

      bool d_enthalpySolve;
      // Pressure Eqn Solver
      PressureSolver* d_pressSolver;
      // Momentum Eqn Solver 
      MomentumSolver* d_momSolver;
      // Scalar solver
      ScalarSolver* d_scalarSolver;
      // physcial constatns
      PhysicalConstants* d_physicalConsts;

}; // End class PicardNonlinearSolver
} // End namespace Uintah


#endif


