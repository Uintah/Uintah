
#ifndef Packages_Uintah_CCA_Components_Examples_SimpleCFD_h
#define Packages_Uintah_CCA_Components_Examples_SimpleCFD_h

#include <Packages/Uintah/CCA/Components/Examples/BoundaryConditions.h>
#include <Packages/Uintah/CCA/Components/Examples/RegionDB.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Core/Geometry/IntVector.h>

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

  class SimpleCFD : public UintahParallelComponent, public SimulationInterface {
  public:
    SimpleCFD(const ProcessorGroup* myworld);
    virtual ~SimpleCFD();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      SimulationStateP&);
    virtual void scheduleInitialize(const LevelP& level,
				    SchedulerP& sched);
    virtual void scheduleComputeStableTimestep(const LevelP& level,
					       SchedulerP&);
    virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);
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
			 DataWarehouse* old_dw, DataWarehouse* new_dw);
    void applyProjection(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw, DataWarehouse* new_dw);

    void advectScalars(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw);
    void diffuseScalar(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw,
		       string varname, const VarLabel* scalar,
		       const VarLabel* scalar_matrix,
		       const VarLabel* scalar_rhs,
		       double rate);
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
		Condition<double>* cbc);
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

    ExamplesLabel* lb_;
    SimulationStateP sharedState_;
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
    IntVector pin_;
    int maxadvect_;
    SimpleMaterial* mymat_;
    double buoyancy_;
    double vorticity_confinement_scale;
    bool keep_pressure;
    bool old_initial_guess;
    bool do_thermal;
    Vector random_initial_velocities;

    RegionDB regiondb;
    BoundaryConditions bcs;
    InitialConditions ics;

    SimpleCFD(const SimpleCFD&);
    SimpleCFD& operator=(const SimpleCFD&);
  };
}

#endif
