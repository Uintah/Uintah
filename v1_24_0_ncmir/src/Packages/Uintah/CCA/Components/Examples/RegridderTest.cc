#include <Packages/Uintah/CCA/Components/Examples/RegridderTest.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

using SCIRun::Point;
using SCIRun::Vector;

namespace Uintah
{
  RegridderTest::RegridderTest ( const ProcessorGroup* myworld ): UintahParallelComponent( myworld )
  {
    d_examplesLabel = scinew ExamplesLabel();
  }
  
  RegridderTest::~RegridderTest ( void )
  {
    delete d_examplesLabel;
  }

  // Interface inherited from Simulation Interface
  void RegridderTest::problemSetup ( const ProblemSpecP& params, GridP& grid, SimulationStateP& state )
  {
    d_sharedState = state;
    d_material = new SimpleMaterial();
    d_sharedState->registerSimpleMaterial( d_material );
  }

  void RegridderTest::scheduleInitialize ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "initialize", this, &RegridderTest::initialize );
    task->computes( d_examplesLabel->density );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "computeStableTimestep", this, &RegridderTest::computeStableTimestep );
    task->computes( d_sharedState->get_delt_label() );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleTimeAdvance ( const LevelP& level, SchedulerP& scheduler, int step, int nsteps )
  {
    Task* task = scinew Task( "timeAdvance", this, &RegridderTest::timeAdvance );
    task->requires( Task::OldDW, d_examplesLabel->density, Ghost::AroundCells, 1 );
    task->computes( d_examplesLabel->density );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "errorEstimate", this, &RegridderTest::errorEstimate, false );
    task->requires( Task::NewDW, d_examplesLabel->density, Ghost::None, 0 );
    task->modifies( d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleInitialErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "initialErrorEstimate", this, &RegridderTest::errorEstimate, true );
    task->requires( Task::NewDW, d_examplesLabel->density, Ghost::None, 0 );
    task->modifies( d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleCoarsen ( const LevelP& level, SchedulerP& scheduler )
  {

  }

  void RegridderTest::scheduleRefine ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "refine", this, &RegridderTest::refine );
    task->requires(Task::NewDW, d_examplesLabel->density, 0, Task::CoarseLevel, 0,
		   Task::NormalDomain, Ghost::None, 0);
    task->computes(d_examplesLabel->density);
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleRefineInterface ( const LevelP& level, SchedulerP& scheduler, int step, int nsteps )
  {
  }

  void RegridderTest::initialize (const ProcessorGroup*,
				  const PatchSubset* patches, const MaterialSubset* matls,
				  DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    //    cerr << "RANDY: RegridderTest::initialize()" << endl;
    BBox gridBoundingBox;
    new_dw->getGrid()->getSpatialRange( gridBoundingBox );
    Vector gridMax( gridBoundingBox.max() );
    Vector gridMin( gridBoundingBox.min() );

    double pi = 3.141592653589;
    d_centerOfDomain   = (( gridMax - gridMin ) / 2.0 ) + gridMin;
    d_radiusOfBall     = 0.10 * gridMax.x();
    d_radiusOfOrbit    = 0.25 * gridMax.x();
    d_currentAngle     = 0;
    d_angularVelocity  = 1;
    d_centerOfBall     = d_centerOfDomain;
    d_centerOfBall[0] += d_radiusOfOrbit * cos( ( pi * d_currentAngle ) / 180.0 );
    d_centerOfBall[1] += d_radiusOfOrbit * sin( ( pi * d_currentAngle ) / 180.0 );
    d_oldCenterOfBall  = d_centerOfBall;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_examplesLabel->density, matl, patch);
	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  density[idx] = distanceToCenterOfDomain.length();
	}
      }
    }
  }


  void RegridderTest::computeStableTimestep (const ProcessorGroup*,
					     const PatchSubset* patches,
					     const MaterialSubset* matls,
					     DataWarehouse* old_dw, DataWarehouse* new_dw)
  {
    new_dw->put(delt_vartype(1), d_sharedState->get_delt_label());
  }

  void RegridderTest::timeAdvance ( const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* old_dw, DataWarehouse* new_dw )
  {
    //    cerr << "RANDY: RegridderTest::timeAdvance()" << endl;
    
    const Level* level = getLevel(patches);
    if ( level->getIndex() == 0 ) {
    }

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_examplesLabel->density, matl, patch);
	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  density[idx] = distanceToCenterOfDomain.length();
	}
      }
    }
  }

  void RegridderTest::errorEstimate ( const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse*, DataWarehouse* new_dw, bool initial )
  {
    double pi = 3.141592653589;
    if ( getLevel(patches)->getIndex() == 0 ) {
      d_currentAngle += d_angularVelocity;
      d_oldCenterOfBall = d_centerOfBall;
      cerr << "RANDY: RegridderTest::scheduleErrorEstimate() center = " << d_centerOfBall << endl;
      d_centerOfBall = d_centerOfDomain;
      d_centerOfBall[0] += d_radiusOfOrbit * cos( ( pi * d_currentAngle ) / 180.0 );
      d_centerOfBall[1] += d_radiusOfOrbit * sin( ( pi * d_currentAngle ) / 180.0 );
      cerr << "RANDY: RegridderTest::scheduleErrorEstimate() after  = " << d_centerOfBall << endl;
    }

    //    cerr << "RANDY: RegridderTest::errorEstimate()" << endl;
    for ( int p = 0; p < patches->size(); p++ ) {
      const Patch* patch = patches->get(p);

      CCVariable<int> refineFlag;
      new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch);

      CCVariable<int> oldRefineFlag;
      new_dw->getModifiable(oldRefineFlag, d_sharedState->get_oldRefineFlag_label(), 0, patch);

      PerPatch<PatchFlagP> refinePatchFlag;
      new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(), 0, patch);
      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

      bool foundErrorOnPatch = false;

      for(int m = 0;m<matls->size();m++) {
	int matl = matls->get(m);
	constCCVariable<double> density;
	new_dw->get( density, d_examplesLabel->density, matl, patch, Ghost::AroundCells, 1 );

	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);

	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector vectorToCenterOfBall = whereThisCellIs - d_centerOfBall;
	  Vector vectorToCenterOfBallOld = whereThisCellIs - d_oldCenterOfBall;

	  if ( vectorToCenterOfBall.length() <= d_radiusOfBall ) {
	    refineFlag[idx]=true;
	    foundErrorOnPatch = true;
	  } else {
	    refineFlag[idx]=false;
	  }
	  if ( vectorToCenterOfBallOld.length() <= d_radiusOfBall ) {
	    oldRefineFlag[idx]=true;
	  } else {
	    oldRefineFlag[idx]=false;
	  }
	}
      }

      if ( foundErrorOnPatch ) {
	refinePatch->flag = true;
      } else {
	refinePatch->flag = false;
      }
    }
  }

  void RegridderTest::refine ( const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse*, DataWarehouse* new_dw )
  {
    //    cerr << "RANDY: RegridderTest::refine()" << endl;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_examplesLabel->density, matl, patch);
	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  density[idx] = distanceToCenterOfDomain.length();
	}
      }
    }
  }
} // namespace Uintah
