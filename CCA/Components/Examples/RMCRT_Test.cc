/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Components/Examples/RMCRT_Test.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>

#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>


using SCIRun::Point;
using SCIRun::Vector;
using SCIRun::DebugStream;

static DebugStream dbg("RMCRT_Test", false);

namespace Uintah
{
  RMCRT_Test::RMCRT_Test ( const ProcessorGroup* myworld ): UintahParallelComponent( myworld )
  {
    //d_examplesLabel = scinew ExamplesLabel();
    d_oldcolorLabel     = VarLabel::create("old_color",     CCVariable<double>::getTypeDescription());
    d_colorLabel        = VarLabel::create("color",         CCVariable<double>::getTypeDescription());
    d_currentAngleLabel = VarLabel::create( "currentAngle", max_vartype::getTypeDescription() );
    d_gac = Ghost::AroundCells;
    d_gn  = Ghost::None;
  }
  
  RMCRT_Test::~RMCRT_Test ( void )
  {
    dbg << UintahParallelComponent::d_myworld->myrank() << " Doing: RMCRT destructor " << endl;
    //delete d_examplesLabel;
  }

  //______________________________________________________________________
  void RMCRT_Test::problemSetup(const ProblemSpecP& params, 
                                const ProblemSpecP& restart_prob_spec, 
                                GridP& grid, 
                                SimulationStateP& state )
  {
    d_sharedState = state;
    d_material = scinew SimpleMaterial();
    d_sharedState->registerSimpleMaterial( d_material );

    ProblemSpecP spec = params->findBlock("RMCRT");
    
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

    spec->get("ballRadius",       d_radiusOfBall);
    spec->get("orbitRadius",      d_radiusOfOrbit);
    spec->get("angularVelocity",  d_angularVelocity);
    spec->get("changingRadius",   d_radiusGrowth);
  }
  
  //______________________________________________________________________
  void RMCRT_Test::scheduleInitialize ( const LevelP& level, 
                                  SchedulerP& scheduler )
  {
    Task* task = scinew Task( "RMCRT_Test::initialize", this, &RMCRT_Test::initialize );
    
    task->computes( d_colorLabel );
    task->computes( d_currentAngleLabel, (Level*)0 );
    
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }

  //______________________________________________________________________
  void RMCRT_Test::scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler )
  {
    printSchedule(level,dbg,"RMCRT_Test::scheduleComputeStableTimestep");
   
    Task* task = scinew Task( "RMCRT_Test::computeStableTimestep", this, &RMCRT_Test::computeStableTimestep );
    
    task->computes( d_sharedState->get_delt_label(),level.get_rep() );
    
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleTimeAdvance ( const LevelP& level, SchedulerP& scheduler)
  {
    printSchedule(level,dbg,"RMCRT_Test::scheduleTimeAdvance");
    
    Task* task = scinew Task( "RMCRT_Test::timeAdvance", this, &RMCRT_Test::timeAdvance );
    
    task->requires( Task::OldDW, d_colorLabel, d_gac, 1 );
    task->computes( d_oldcolorLabel );
    task->computes( d_colorLabel );
    
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    printSchedule(level,dbg,"RMCRT_Test::errorEstimate");
    
    Task* task = scinew Task( "RMCRT_Test::errorEstimate", this, &RMCRT_Test::errorEstimate, false );
    
    task->requires( Task::OldDW, d_currentAngleLabel, (Level*) 0);
    task->requires( Task::NewDW, d_colorLabel,    d_gac, 1 );
    task->requires( Task::NewDW, d_oldcolorLabel, d_gac, 1 );
    
    task->modifies( d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(),   d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    task->computes( d_currentAngleLabel, (Level*) 0);
    
    scheduler->addTask( task, scheduler->getLoadBalancer()->getPerProcessorPatchSet(level), d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleInitialErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
  {
    printSchedule(level,dbg,"RMCRT_Test::scheduleInitialErrorEstimate");
    
    Task* task = scinew Task( "RMCRT_Test::initialErrorEstimate", this, &RMCRT_Test::errorEstimate, true );
    
    task->requires( Task::NewDW, d_colorLabel, d_gac, 1 );
    task->modifies( d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_oldRefineFlag_label(),   d_sharedState->refineFlagMaterials() );
    task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );
    
    scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleCoarsen ( const LevelP& coarseLevel, SchedulerP& scheduler )
  {
    printSchedule(coarseLevel,dbg,"RMCRT_Test::scheduleCoarsen");
    
    Task* task = scinew Task( "RMCRT_Test::coarsen", this, &RMCRT_Test::coarsen );
    
    task->requires(Task::NewDW, d_colorLabel, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    task->modifies(d_colorLabel);
    
    scheduler->addTask( task, coarseLevel->eachPatch(), d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleRefine ( const PatchSet* patches, 
                                    SchedulerP& scheduler )
  {
    printSchedule(patches,dbg,"RMCRT_Test::scheduleRefine");
    
    Task* task = scinew Task( "RMCRT_Test::refine", this, &RMCRT_Test::refine );
    
    task->requires(Task::NewDW, d_colorLabel, 0, Task::CoarseLevel, 0, Task::NormalDomain, d_gn, 0);
    //    task->requires(Task::NewDW, d_oldcolorLabel, 0, Task::CoarseLevel, 0,
    //		   Task::NormalDomain, d_gn, 0);
    scheduler->addTask( task, patches, d_sharedState->allMaterials() );
  }
  //______________________________________________________________________
  void RMCRT_Test::scheduleRefineInterface ( const LevelP&, 
                                             SchedulerP&, 
                                             bool, 
                                             bool)
  {
  }
  //______________________________________________________________________
  void RMCRT_Test::initialize (const ProcessorGroup*,
			          const PatchSubset* patches, 
                               const MaterialSubset* matls,
			          DataWarehouse*, 
                               DataWarehouse* new_dw)
  {
    new_dw->put(max_vartype(0), d_currentAngleLabel);
    d_centerOfBall     = d_centerOfDomain;
    d_centerOfBall[0] += d_radiusOfOrbit; // *cos(0)
    d_oldCenterOfBall  = d_centerOfBall;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	
       CCVariable<double> color;
	new_dw->allocateAndPut(color, d_colorLabel, matl, patch);
	
       for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	
         IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  color[idx] = distanceToCenterOfDomain.length();
	}
      }
    }
  }


  //______________________________________________________________________
  void RMCRT_Test::computeStableTimestep (const ProcessorGroup*,
					       const PatchSubset* patches,                          
					       const MaterialSubset* /*matls*/,                     
					       DataWarehouse* /*old_dw*/, 
                                          DataWarehouse* new_dw)    
  {
    const Level* level = getLevel(patches);
    double delt = level->dCell().x();
    new_dw->put(delt_vartype(delt), d_sharedState->get_delt_label(), level);
  }
  //______________________________________________________________________
  void RMCRT_Test::timeAdvance ( const ProcessorGroup*,
			           const PatchSubset* patches,                        
			           const MaterialSubset* matls,                       
			           DataWarehouse* old_dw, 
                                DataWarehouse* new_dw )     
  { 
    const Level* level = getLevel(patches);
    if ( level->getIndex() == 0 ) {
    }

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      printTask(patches, patch,dbg,"Doing timeAdvance");
      
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	
       CCVariable<double> color;
	new_dw->allocateAndPut(color, d_colorLabel, matl, patch);

        // an exercise to get from the old and put in the new (via mpi amr)...
	constCCVariable<double> oldDWcolor;
       CCVariable<double> oldcolor;
	
       old_dw->get(           oldDWcolor, d_colorLabel,    matl, patch, d_gac, 1 );
       new_dw->allocateAndPut(oldcolor,   d_oldcolorLabel, matl, patch);


	for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
         
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  
         color[idx] = distanceToCenterOfDomain.length();
         oldcolor[idx] = oldDWcolor[idx];
	}
      }
    }
  }
  //______________________________________________________________________
  void RMCRT_Test::errorEstimate ( const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* matls,
				      DataWarehouse* old_dw, 
                                  DataWarehouse* new_dw, 
                                  bool initial )
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
          
      //cerr << "RANDY: RMCRT_Test::scheduleErrorEstimate() after  = " << d_centerOfBall << endl;
    }

    for ( int p = 0; p < patches->size(); p++ ) {
      const Patch* patch = patches->get(p);
      
      printTask(patches, patch,dbg,"Doing errorEstimate");

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
	 constCCVariable<double> color;
	 constCCVariable<double> oldcolor;
	 new_dw->get( color, d_colorLabel, matl, patch, d_gac, 1 );
        
        if (!initial){
          new_dw->get( oldcolor, d_oldcolorLabel, matl, patch, d_gac, 1 );
        }
    
	 for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	   IntVector idx(*iter);

	   if ( color[idx] <= d_radiusOfBall ) {
	     refineFlag[idx]=true;
	     foundErrorOnPatch = true;
	   } else {
	     refineFlag[idx]=false;
	   }
           if (!initial) {
             if ( oldcolor[idx] <= d_radiusOfBall ) {
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
  //______________________________________________________________________
  void RMCRT_Test::coarsen ( const ProcessorGroup*,
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
      
      printTask(patches, coarsePatch,dbg,"Doing coarsen");
    
      // Find the overlapping regions...
      Level::selectType finePatches;
      coarsePatch->getFineLevelPatches(finePatches);
      
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);
        //__________________________________
        //   D E N S I T Y
        CCVariable<double> color;
        new_dw->getModifiable(color, d_colorLabel, matl, coarsePatch);
        //print(color, "before coarsen color");
        
        for(int i=0;i<finePatches.size();i++){
          const Patch* finePatch = finePatches[i];
    
          constCCVariable<double> fine_den;
          new_dw->get(fine_den, d_colorLabel, matl, finePatch,d_gn, 0);
          
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
            color[*iter]=rho_tmp*ratio;
          }
        }  // fine patch loop
      }
    }  // course patch loop 
  }

  //______________________________________________________________________
  void RMCRT_Test::refine ( const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* matls,
			       DataWarehouse*, DataWarehouse* new_dw )
  {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      
      printTask(patches, patch,dbg,"Doing refine");
   
      for(int m = 0;m<matls->size();m++){
	int matl = matls->get(m);
	
       CCVariable<double> color;
	new_dw->allocateAndPut(color, d_colorLabel, matl, patch);
	
       for ( CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
	  IntVector idx(*iter);
	  Vector whereThisCellIs( patch->cellPosition( idx ) );
	  Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
	  color[idx] = distanceToCenterOfDomain.length();
	}
      }
    }
  }
//__________________________________
//  
void RMCRT_Test::printSchedule(const PatchSet* patches,
                              DebugStream& dbg,
                              const string& where)
{
  if (dbg.active()){
    dbg << UintahParallelComponent::d_myworld->myrank() << " ";
    dbg << left;
    dbg.width(50);
    dbg  << where << "L-"
        << getLevel(patches)->getIndex()<< endl;
  }  
}
//__________________________________
//
void RMCRT_Test::printSchedule(const LevelP& level,
                              DebugStream& dbg,
                              const string& where)
{
  if (dbg.active()){
    dbg << UintahParallelComponent::d_myworld->myrank() << " ";
    dbg << left;
    dbg.width(50);
    dbg << where << "L-"
        << level->getIndex()<< endl;
  }  
}
//__________________________________
//
void RMCRT_Test::printTask(const PatchSubset* patches,
                          const Patch* patch,
                          DebugStream& dbg,
                          const string& where)
{
  if (dbg.active()){
    dbg << UintahParallelComponent::d_myworld->myrank() << " ";
    dbg << left;
    dbg.width(50);
    dbg << where << " MPM \tL-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex()<< endl;
  }  
}

//__________________________________
//
void RMCRT_Test::printTask(const Patch* patch,
                          DebugStream& dbg,
                          const string& where)
{
  if (dbg.active()){
    dbg << UintahParallelComponent::d_myworld->myrank() << " ";
    dbg << left;
    dbg.width(50);
    dbg << where << " MPM \tL-"
        << patch->getLevel()->getIndex()
        << " patch " << patch->getGridIndex()<< endl;
  }  
}  
 
} // namespace Uintah
