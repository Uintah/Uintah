#include <Packages/Uintah/CCA/Components/Examples/RegridderTest.h>
#include <Packages/Uintah/CCA/Components/Examples/ExamplesLabel.h>
#include <Packages/Uintah/CCA/Components/Regridder/PerPatchVars.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Util/DebugStream.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

using SCIRun::Point;
using SCIRun::Vector;
using SCIRun::DebugStream;

static DebugStream dbg("RegridderTest", false);

namespace Uintah
{
  RegridderTest::RegridderTest ( const ProcessorGroup* myworld ): UintahParallelComponent( myworld )
  {
    //d_examplesLabel = scinew ExamplesLabel();
    d_oldDensityLabel = VarLabel::create("old_density",
                                         CCVariable<double>::getTypeDescription());
    d_densityLabel = VarLabel::create("density",
                                      CCVariable<double>::getTypeDescription());

    d_currentAngleLabel = VarLabel::create( "currentAngle", max_vartype::getTypeDescription() );
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

    ProblemSpecP spec = params->findBlock("RegridderTest");
    
    BBox gridBoundingBox;
    grid->getSpatialRange( gridBoundingBox );
    d_gridMax = gridBoundingBox.max().asVector();
    d_gridMin = gridBoundingBox.min().asVector();
    d_centerOfDomain   = (( d_gridMax - d_gridMin ) / 2.0 ) + d_gridMin;

    //defaults
    d_radiusOfBall     = 0.10 * d_gridMax.x();
    d_radiusOfOrbit    = 0.25 * d_gridMax.x();
    d_angularVelocity  = 10;

    d_radiusGrowth = false;
    d_radiusGrowthDir = true; // true is to expand, false to shrink

    spec->get("ballRadius", d_radiusOfBall);
    spec->get("orbitRadius", d_radiusOfOrbit);
    spec->get("angularVelocity", d_angularVelocity);
    spec->get("changingRadius", d_radiusGrowth);

  }

  void RegridderTest::scheduleInitialize ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "initialize", this, &RegridderTest::initialize );
    task->computes( d_densityLabel );
    task->computes( d_currentAngleLabel, (Level*)0 );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "computeStableTimestep", this, &RegridderTest::computeStableTimestep );
    task->computes( d_sharedState->get_delt_label() );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleTimeAdvance ( const LevelP& level, SchedulerP& scheduler, int /*step*/, int /*nsteps*/ )
  {
    Task* task = scinew Task( "timeAdvance", this, &RegridderTest::timeAdvance );
    task->requires( Task::OldDW, d_densityLabel, Ghost::AroundCells, 1 );
    task->computes( d_oldDensityLabel );
    task->computes( d_densityLabel );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "errorEstimate", this, &RegridderTest::errorEstimate, false );
    task->requires( Task::OldDW, d_currentAngleLabel, (Level*) 0);
    task->requires( Task::NewDW, d_densityLabel, Ghost::AroundCells, 1 );
    task->requires( Task::NewDW, d_oldDensityLabel, Ghost::AroundCells, 1 );
    task->modifies( d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    task->computes( d_currentAngleLabel, (Level*) 0);
    scheduler->addTask( task, scheduler->getLoadBalancer()->createPerProcessorPatchSet(level), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleInitialErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "initialErrorEstimate", this, &RegridderTest::errorEstimate, true );
    task->requires( Task::NewDW, d_densityLabel, Ghost::AroundCells, 1 );
    task->modifies( d_sharedState->get_refineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(), d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleCoarsen ( const LevelP& coarseLevel, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "coarsen", this, &RegridderTest::coarsen );
    task->requires(Task::NewDW, d_densityLabel,
                   0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0);
    task->modifies(d_densityLabel);
    scheduler->addTask( task, coarseLevel->eachPatch(), d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleRefine ( const PatchSet* patches, SchedulerP& scheduler )
  {
    Task* task = scinew Task( "refine", this, &RegridderTest::refine );
    task->requires(Task::NewDW, d_densityLabel, 0, Task::CoarseLevel, 0,
		   Task::NormalDomain, Ghost::None, 0);
    //    task->requires(Task::NewDW, d_oldDensityLabel, 0, Task::CoarseLevel, 0,
    //		   Task::NormalDomain, Ghost::None, 0);
    scheduler->addTask( task, patches, d_sharedState->allMaterials() );
  }

  void RegridderTest::scheduleRefineInterface ( const LevelP& /*level*/, SchedulerP& /*scheduler*/, 
                                                int /*step*/, int /*nsteps*/ )
  {
  }

  void RegridderTest::initialize (const ProcessorGroup*,
				  const PatchSubset* patches, const MaterialSubset* matls,
				  DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
  {
    //    cerr << "RANDY: RegridderTest::initialize()" << endl;
    
    new_dw->put(max_vartype(0), d_currentAngleLabel);
    d_centerOfBall     = d_centerOfDomain;
    d_centerOfBall[0] += d_radiusOfOrbit; // *cos(0)
    d_oldCenterOfBall  = d_centerOfBall;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_densityLabel, matl, patch);
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
					     const MaterialSubset* /*matls*/,
					     DataWarehouse* /*old_dw*/, DataWarehouse* new_dw)
  {
    const Level* level = getLevel(patches);
    double delt = level->dCell().x();
    delt = level->adjustDelt(delt);
    new_dw->put(delt_vartype(delt), d_sharedState->get_delt_label());
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
      dbg << d_myworld->myrank() << "  RegridderTest::timeAdvance() on patch " << patch->getID()<< endl;
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_densityLabel, matl, patch);

        // an exercise to get from the old and put in the new (via mpi amr)...
	constCCVariable<double> oldDWDensity;
        CCVariable<double> oldDensity;
	old_dw->get( oldDWDensity, d_densityLabel, matl, patch, Ghost::AroundCells, 1 );
        new_dw->allocateAndPut(oldDensity, d_oldDensityLabel, matl, patch);


	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  density[idx] = distanceToCenterOfDomain.length();
          oldDensity[idx] = oldDWDensity[idx];
	}
      }
    }
  }

  void RegridderTest::errorEstimate ( const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw, DataWarehouse* new_dw, bool initial )
  {
    
    double pi = 3.141592653589;
    if ( getLevel(patches)->getIndex() == getLevel(patches)->getGrid()->numLevels()-1 ) {
      max_vartype angle;
      double currentAngle;
      if (!initial) {
        old_dw->get(angle, d_currentAngleLabel);
        currentAngle = angle + d_angularVelocity;
        new_dw->put(max_vartype(currentAngle), d_currentAngleLabel);
      }
      else {
        currentAngle = angle;
      }
        
      d_oldCenterOfBall = d_centerOfBall;
      // cerr << "RANDY: RegridderTest::scheduleErrorEstimate() center = " << d_centerOfBall << endl;
      d_centerOfBall = d_centerOfDomain;
      d_centerOfBall[0] += d_radiusOfOrbit * cos( ( pi * currentAngle ) / 180.0 );
      d_centerOfBall[1] += d_radiusOfOrbit * sin( ( pi * currentAngle ) / 180.0 );

      if (d_radiusGrowth) {
        if (d_radiusGrowthDir) {
          d_radiusOfBall     += 0.001 * d_gridMax.x();
          if (d_radiusOfBall > .25 * d_gridMax.x())
            d_radiusGrowthDir = false;
        }
        else {
          d_radiusOfBall     -= 0.001 * d_gridMax.x();
          if (d_radiusOfBall < .1 * d_gridMax.x())
            d_radiusGrowthDir = true;
        }
      }
          
      //cerr << "RANDY: RegridderTest::scheduleErrorEstimate() after  = " << d_centerOfBall << endl;
    }

    for ( int p = 0; p < patches->size(); p++ ) {
      const Patch* patch = patches->get(p);
      dbg << d_myworld->myrank() << "  RegridderTest::errorEstimate() on patch " << patch->getID()<< endl;

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
	constCCVariable<double> oldDensity;
	new_dw->get( density, d_densityLabel, matl, patch, Ghost::AroundCells, 1 );
        if (!initial)
          new_dw->get( oldDensity, d_oldDensityLabel, matl, patch, Ghost::AroundCells, 1 );

	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);

	  if ( density[idx] <= d_radiusOfBall ) {
	    refineFlag[idx]=true;
	    foundErrorOnPatch = true;
	  } else {
	    refineFlag[idx]=false;
	  }
          if (!initial) {
            if ( oldDensity[idx] <= d_radiusOfBall ) {
              oldRefineFlag[idx]=true;
            } else {
              oldRefineFlag[idx]=false;
            }
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

  void RegridderTest::coarsen ( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse*, DataWarehouse* new_dw )
  {
    const Level* coarseLevel = getLevel(patches);
    const LevelP fineLevel = coarseLevel->getFinerLevel();
    IntVector rr(fineLevel->getRefinementRatio());
    double ratio = 1./(rr.x()*rr.y()*rr.z());
    for(int p=0;p<patches->size();p++){  
      const Patch* coarsePatch = patches->get(p);
      dbg << d_myworld->myrank() << "  RegridderTest::coarsen() on patch " << coarsePatch->getID()<< endl;
      // Find the overlapping regions...
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        //__________________________________
        //   D E N S I T Y
        CCVariable<double> density;
        new_dw->getModifiable(density, d_densityLabel, matl, coarsePatch);
        //print(density, "before coarsen density");
        
        for(int i=0;i<finePatches.size();i++){
          const Patch* finePatch = finePatches[i];
          constCCVariable<double> fine_den;
          new_dw->get(fine_den, d_densityLabel, matl, finePatch,
                      Ghost::None, 0);
          
          IntVector fl(finePatch->getCellLowIndex());
          IntVector fh(finePatch->getCellHighIndex());
          IntVector l(fineLevel->mapCellToCoarser(fl));
          IntVector h(fineLevel->mapCellToCoarser(fh));
          l = Max(l, coarsePatch->getCellLowIndex());
          h = Min(h, coarsePatch->getCellHighIndex());
          
          for(CellIterator iter(l, h); !iter.done(); iter++){
            double rho_tmp=0;
            IntVector fineStart(coarseLevel->mapCellToFiner(*iter));
            
            for(CellIterator inside(IntVector(0,0,0), fineLevel->getRefinementRatio());
                !inside.done(); inside++){
              rho_tmp+=fine_den[fineStart+*inside];
            }
            density[*iter]=rho_tmp*ratio;
          }
        }  // fine patch loop
      }
    }  // course patch loop 
  }


  void RegridderTest::refine ( const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse*, DataWarehouse* new_dw )
  {
    //    cerr << "RANDY: RegridderTest::refine()" << endl;
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      dbg << d_myworld->myrank() << "  RegridderTest::refine() on patch " << patch->getID()<< endl;
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	CCVariable<double> density;
	new_dw->allocateAndPut(density, d_densityLabel, matl, patch);
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
