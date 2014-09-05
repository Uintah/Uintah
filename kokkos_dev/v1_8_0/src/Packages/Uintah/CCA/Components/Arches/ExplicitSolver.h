//----- ExplicitSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExplicitSolver_h
#define Uintah_Component_Arches_ExplicitSolver_h

/**************************************
CLASS
   NonlinearSolver
   
   Class ExplicitSolver is a subclass of NonlinearSolver
   which implements the Forward Euler/RK2/ RK3 methods

GENERAL INFORMATION
   ExplicitSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000


KEYWORDS


DESCRIPTION
   Class ExplicitSolver implements the
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
class ReactiveScalarSolver; 
class TurbulenceModel;
class Properties;
class BoundaryCondition;
class PhysicalConstants;
class EnthalpySolver;
class ExplicitSolver: public NonlinearSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Solver initialized with all input data 
      ExplicitSolver(const ArchesLabel* label,
		     const MPMArchesLabel* MAlb,
		     Properties* props, 
		     BoundaryCondition* bc,
		     TurbulenceModel* turbModel, 
		     PhysicalConstants* physConst,
		     bool calcReactscalar,
		     bool calcEnthalpy,
		     const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for ExplicitSolver.
      virtual ~ExplicitSolver();


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
      // Do not solve the nonlinear system but just copy variables to end
      // so that they retain the right guess for the next step

      virtual int noSolve(const LevelP& level,
			  SchedulerP& sched);
  
      ///////////////////////////////////////////////////////////////////////
      // Schedule the Initialization of non linear solver
      //    [in] 
      //        data User data needed for solve 
      void sched_setInitialGuess(SchedulerP&, const PatchSet* patches,
				 const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule dummy solve (data copy) for first time step of MPMArches
      // to overcome scheduler limitation on getting pset from old_dw

      void sched_dummySolve(SchedulerP& sched,
			    const PatchSet* patches,
			    const MaterialSet* matls);

      ///////////////////////////////////////////////////////////////////////
      // Schedule the interpolation of velocities from Face Centered Variables
      //    to a Cell Centered Vector
      //    [in] 
      void sched_interpolateFromFCToCC(SchedulerP&, const PatchSet* patches,
				       const MaterialSet* matls);

      void sched_interpolateFromFCToCCPred(SchedulerP&, const PatchSet* patches,
				       const MaterialSet* matls);

      void sched_interpolateFromFCToCCInterm(SchedulerP&, const PatchSet* patches,
				       const MaterialSet* matls);

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

      void sched_printTotalKE(SchedulerP& sched,
			      const PatchSet* patches,
			      const MaterialSet* matls,
			      const int Runge_Kutta_current_step,
			      const bool Runge_Kutta_last_step);
  
protected :

private:

      // GROUP: Constructors (private):
      ////////////////////////////////////////////////////////////////////////
      // Should never be used
      ExplicitSolver();

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
      // actual data copy for first time step of MPMArches to overcome
      // scheduler limitation on getting pset from old_dw

      void dummySolve(const ProcessorGroup* pc,
		      const PatchSubset* patches,
		      const MaterialSubset* matls,
		      DataWarehouse* old_dw,
		      DataWarehouse* new_dw);

      ///////////////////////////////////////////////////////////////////////
      // Actually Interpolate from SFCX, SFCY, SFCZ to CC<Vector>
      //    [in] 
      void interpolateFromFCToCC(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);

      void interpolateFromFCToCCPred(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);

      void interpolateFromFCToCCInterm(const ProcessorGroup* pc,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);

      void probeData(const ProcessorGroup* pc,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);

      void printTotalKE(const ProcessorGroup* ,
			const PatchSubset* patches,
			const MaterialSubset*,
			DataWarehouse*,
			DataWarehouse* new_dw,
			const int Runge_Kutta_current_step,
			const bool Runge_Kutta_last_step);

private:
      // const VarLabel*
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      // generation variable for DataWarehouse creation

      // Total number of nonlinear iterates
      int d_nonlinear_its;
      // for probing data for debuging or plotting
      bool d_probe_data;
      // properties...solves density, temperature and specie concentrations
      Properties* d_props;
      // Boundary conditions
      BoundaryCondition* d_boundaryCondition;
      // Turbulence Model
      TurbulenceModel* d_turbModel;
      bool d_reactingScalarSolve;
      bool d_enthalpySolve;
      vector<IntVector> d_probePoints;
      // nonlinear residual tolerance
      double d_resTol;
      // Pressure Eqn Solver
      PressureSolver* d_pressSolver;
      // Momentum Eqn Solver 
      MomentumSolver* d_momSolver;
      // Scalar solver
      ScalarSolver* d_scalarSolver;
      // reacting scalar solver
      ReactiveScalarSolver* d_reactingScalarSolver;
      // enthalpy solver
      EnthalpySolver* d_enthalpySolver;
      // physcial constatns
      PhysicalConstants* d_physicalConsts;

}; // End class ExplicitSolver
} // End namespace Uintah


#endif


