
#ifndef Packages_Uintah_CCA_Components_Examples_SimpleCFD_h
#define Packages_Uintah_CCA_Components_Examples_SimpleCFD_h

#include <Core/Grid/Variables/Stencil7.h>
#include <CCA/Components/Examples/BoundaryConditions.h>
#include <CCA/Components/Examples/RegionDB.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Task.h>
#include <SCIRun/Core/Geometry/IntVector.h>

#include <CCA/Components/Examples/uintahshare.h>
namespace Uintah {
  class SimpleMaterial;
  class ExamplesLabel;
  class SolverInterface;
  class SolverParameters;

/**************************************

CLASS
   SimpleCFD
   
   SimpleCFD simulation

GENERAL INFORMATION

   SimpleCFD.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SimpleCFD

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UINTAHSHARE SimpleCFD : public UintahParallelComponent, public SimulationInterface {
  public:
    SimpleCFD(const ProcessorGroup* myworld);
    virtual ~SimpleCFD();

    virtual void problemSetup(const ProblemSpecP& params, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);
				      

    struct DiffuseInfo {
      string varname;
      const VarLabel* scalar;
      const VarLabel* scalar_matrix;
      const VarLabel* scalar_rhs;
      double rate;
      DiffuseInfo(const string& varname, const VarLabel* scalar,
		  const VarLabel* scalar_matrix, const VarLabel* scalar_rhs,
		  double rate)
	: varname(varname), scalar(scalar), scalar_matrix(scalar_matrix),
	scalar_rhs(scalar_rhs), rate(rate)
	{
	}
    };
  protected:
    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
    bool do_thermal;
    bool keep_pressure;

    virtual void refineBoundaries(const Patch* patch,
				  CCVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCXVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCYVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void refineBoundaries(const Patch* patch,
				  SFCZVariable<double>& val,
				  DataWarehouse* new_dw,
				  const VarLabel* label,
				  int matl, double factor);
    virtual void addRefineDependencies(Task* task, const VarLabel* var,
				       bool needCoarseOld, bool needCoarseNew);
    void schedulePressureSolve(const LevelP& level, SchedulerP& sched,
			       SolverInterface* solver,
			       const VarLabel* pressure,
			       const VarLabel* pressure_matrix,
			       const VarLabel* pressure_rhs,
                               bool usePressureAsGuess=true);
  private:
    void scheduleDiffuseScalar(SchedulerP& sched, const LevelP& level,
			       const string& name,
			       const VarLabel* scalar,
			       const VarLabel* scalar_matrix,
			       const VarLabel* scalar_rhs,
			       double rate,
			       SolverInterface* solver,
			       const SolverParameters* solverparams);

    void initialize(const ProcessorGroup*,
		    const PatchSubset* patches, const MaterialSubset* matls,
		    DataWarehouse* old_dw, DataWarehouse* new_dw);
    void hackbcs(const ProcessorGroup*,
		 const PatchSubset* patches, const MaterialSubset* matls,
		 DataWarehouse* old_dw, DataWarehouse* new_dw);
    void computeStableTimestep(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);

    void advectVelocity(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw);
    void applyForces(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset* matls,
		     DataWarehouse* old_dw, DataWarehouse* new_dw);
    void applyViscosity(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw, DataWarehouse* new_dw,
			int dir);
    void projectVelocity(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw, DataWarehouse* new_dw,
			 const VarLabel* pressure,
			 const VarLabel* pressure_matrix,
			 const VarLabel* pressure_rhs);
    void applyProjection(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw, DataWarehouse* new_dw,
			 const VarLabel* pressure);

    void advectScalars(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void diffuseScalar(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw,
		       DiffuseInfo di);
    void dissipateScalars(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw, DataWarehouse* new_dw);

    void updatebcs(const ProcessorGroup*,
		   const PatchSubset* patches,
		   const MaterialSubset* matls,
		   DataWarehouse* old_dw, DataWarehouse* new_dw);
    void interpolateVelocities(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse* old_dw, DataWarehouse* new_dw);

    void advect(Array3<double>& q, const Array3<double>& qold,
		CellIterator iter,
		const Patch* patch, double delt, const Vector& offset,
		constSFCXVariable<double>& xvel,
		constSFCYVariable<double>& yvel,
		constSFCZVariable<double>& zvel,
		constNCVariable<int>& bctype,
		Condition<double>* cbc,
		Condition<double>* xbc,
		Condition<double>* ybc,
		Condition<double>* zbc);
    void applybc(const IntVector& idx, const IntVector& l, const IntVector& h,
		 const IntVector& h2,
		 const Array3<double>& field, double delt,
		 const Vector& inv_dx2, double diff,
		 constNCVariable<int>& bctype, Condition<double>* scalar_bc,
		 Condition<double>* xflux_bc, Condition<double>* yflux_bc,
		 Condition<double>* zflux_bc,
		 const IntVector& FW, const IntVector& FE,
		 const IntVector& FS, const IntVector& FN,
		 const IntVector& FB, const IntVector& FT,
		 Array3<Stencil7>& A, Array3<double>& rhs);

    double delt_multiplier_;
    double density_diffusion_;
    SolverParameters* diffusion_params_;
    double density_dissipation_;
    double thermal_conduction_;
    SolverParameters* conduction_params_;
    double viscosity_;
    SolverParameters* viscosity_params_;
    SolverParameters* pressure_params_;
    double advection_tolerance_;
    Point pressure_pin_;
    int maxadvect_;
    SimpleMaterial* mymat_;
    double buoyancy_;
    double vorticity_confinement_scale_;
    bool old_initial_guess;
    Vector random_initial_velocities;
    
    RegionDB regiondb;
    InitialConditions ics;

    SimpleCFD(const SimpleCFD&);
    SimpleCFD& operator=(const SimpleCFD&);

  protected:
    BoundaryConditions bcs;
  };
}

#endif
