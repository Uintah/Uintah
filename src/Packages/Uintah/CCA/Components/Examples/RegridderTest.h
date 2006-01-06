#ifndef Packages_Uintah_CCA_Components_Examples_RegridderTest_h
#define Packages_Uintah_CCA_Components_Examples_RegridderTest_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Core/Geometry/Vector.h>

namespace Uintah
{
  class SimpleMaterial;
  class ExamplesLabel;
  class VarLabel;
/**************************************

CLASS
   RegridderTest
   
   RegridderTest simulation

GENERAL INFORMATION

   RegridderTest.h

   Randy N. Jones
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2004 SCI Group

KEYWORDS
   Regridder

DESCRIPTION
   This is a simple component that will test Oren Livne's regridding
   algorith when using AMR.
  
WARNING
  
****************************************/

  class RegridderTest: public UintahParallelComponent, public SimulationInterface {
  public:
    RegridderTest ( const ProcessorGroup* myworld );
    virtual ~RegridderTest ( void );

    // Interface inherited from Simulation Interface
    virtual void problemSetup                  ( const ProblemSpecP& params, GridP& grid, SimulationStateP& state );
    virtual void scheduleInitialize            ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleTimeAdvance           ( const LevelP& level, SchedulerP& scheduler, int step, int nsteps );
    virtual void scheduleErrorEstimate         ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleInitialErrorEstimate  ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleCoarsen               ( const LevelP& level, SchedulerP& scheduler );
    virtual void scheduleRefine                ( const PatchSet* patches, SchedulerP& scheduler );
    virtual void scheduleRefineInterface       ( const LevelP& level, SchedulerP& scheduler, int step, int nsteps );

  private:
    void initialize ( const ProcessorGroup*,
		      const PatchSubset* patches, const MaterialSubset* matls,
		      DataWarehouse* old_dw, DataWarehouse* new_dw );

    void computeStableTimestep ( const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset* matls,
				 DataWarehouse* old_dw, DataWarehouse* new_dw );

    void timeAdvance ( const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset* matls,
		       DataWarehouse* old_dw, DataWarehouse* new_dw);

    void errorEstimate ( const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse*, DataWarehouse* new_dw, bool initial);

    void coarsen ( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse*, DataWarehouse* new_dw);

    void refine ( const ProcessorGroup*,
		  const PatchSubset* patches,
		  const MaterialSubset* matls,
		  DataWarehouse*, DataWarehouse* new_dw);

    ExamplesLabel*   d_examplesLabel;
    SimulationStateP d_sharedState;
    SimpleMaterial*  d_material;

    VarLabel* d_oldDensityLabel;
    VarLabel* d_densityLabel;
    VarLabel* d_currentAngleLabel;
    SCIRun::Vector d_gridMax;
    SCIRun::Vector d_gridMin;

    // Fake cylinder
    SCIRun::Vector d_centerOfBall;
    SCIRun::Vector d_centerOfDomain;
    SCIRun::Vector d_oldCenterOfBall;
    double         d_radiusOfBall;
    double         d_radiusOfOrbit;
    double         d_angularVelocity;

    bool           d_radiusGrowth;
    bool           d_radiusGrowthDir;
  };

} // namespace Uintah

#endif
