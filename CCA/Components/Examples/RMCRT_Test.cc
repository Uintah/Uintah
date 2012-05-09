/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <sci_defs/cuda_defs.h>

using namespace std;
using SCIRun::Point;
using SCIRun::Vector;
using SCIRun::DebugStream;

static DebugStream dbg("RMCRT_Test", false);


namespace Uintah
{
//______________________________________________________________________
//
RMCRT_Test::RMCRT_Test ( const ProcessorGroup* myworld ): UintahParallelComponent( myworld )
{
  d_colorLabel    = VarLabel::create( "color",    CCVariable<double>::getTypeDescription() );          
  d_divQLabel     = VarLabel::create( "divQ",     CCVariable<double>::getTypeDescription() );          
  d_abskgLabel    = VarLabel::create( "abskg",    CCVariable<double>::getTypeDescription() );          
  d_absorpLabel   = VarLabel::create( "absorp",   CCVariable<double>::getTypeDescription() );
  d_sigmaT4Label  = VarLabel::create( "sigmaT4",  CCVariable<double>::getTypeDescription() );
   
  d_gac = Ghost::AroundCells;
  d_gn  = Ghost::None;
  d_matl = 0;
  d_initColor = -9;
  d_initAbskg = -9;
  d_CoarseLevelRMCRTMethod = true;
  d_multiLevelRMCRTMethod  = false;
}
//______________________________________________________________________
//
RMCRT_Test::~RMCRT_Test ( void )
{
  if ( d_RMCRT ) {
    delete d_RMCRT;
  }
    
  VarLabel::destroy(d_colorLabel);
  VarLabel::destroy(d_divQLabel);
  VarLabel::destroy(d_abskgLabel);
  VarLabel::destroy(d_absorpLabel);
  VarLabel::destroy(d_sigmaT4Label);
  
  dbg << UintahParallelComponent::d_myworld->myrank() << " Doing: RMCRT destructor " << endl;

  for (int i = 0; i< (int)d_refine_geom_objs.size(); i++) {
    delete d_refine_geom_objs[i];
  }
  //delete d_examplesLabel;
}

//______________________________________________________________________
void RMCRT_Test::problemSetup(const ProblemSpecP& prob_spec, 
                              const ProblemSpecP& restart_prob_spec, 
                              GridP& grid, 
                              SimulationStateP& state )
{
  d_sharedState = state;
  d_material = scinew SimpleMaterial();
  d_sharedState->registerSimpleMaterial( d_material );


  //manually manipulate the scheduling of copy data for the shootRay task
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
  sched->overrideVariableBehavior("color",false, false, true);

  //__________________________________
  //  RMCRT
  if (prob_spec->findBlock("RMCRT")){
    ProblemSpecP rmcrt_ps = prob_spec->findBlock("RMCRT"); 
    
#ifdef HAVE_CUDA
    d_RMCRT = scinew Ray(dynamic_cast<GPUThreadedMPIScheduler*>(getPort("scheduler")));
#else
    d_RMCRT = scinew Ray();
#endif

    d_RMCRT->registerVarLabels(0,d_abskgLabel,
                                 d_absorpLabel,
                                 d_colorLabel,
                                 d_divQLabel);
                                 
    rmcrt_ps->require("Temperature",  d_initColor);
    rmcrt_ps->require("abskg",        d_initAbskg);
    
    d_RMCRT->problemSetup( prob_spec, rmcrt_ps );
    proc0cout << "__________________________________ Reading in RMCRT section of ups file" << endl;
  }
  //__________________________________
  //  bullet proofing
  IntVector extraCells = grid->getLevel(0)->getExtraCells();
  IntVector periodic   = grid->getLevel(0)->getPeriodicBoundaries();

  for (int dir=0; dir<3; dir++){
    if(periodic[dir] != 1 && extraCells[dir] !=1) {
      ostringstream warn;
      warn<< "\n \n INPUT FILE ERROR: \n You must have either a periodic boundary " << periodic
          << " or one extraCell "<< extraCells << " specified in each direction";

      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
    }
  }
  
  //__________________________________
  //  Read in the AMR section
  ProblemSpecP rmcrt_ps;
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");
  if (amr_ps){
    rmcrt_ps = amr_ps->findBlock("RMCRT");

    if(!rmcrt_ps){
      string warn;
      warn ="\n INPUT FILE ERROR:\n <RMCRT>  block not found inside of <AMR> block \n";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }

    rmcrt_ps->require("CoarseLevelRMCRTMethod", d_CoarseLevelRMCRTMethod);
    rmcrt_ps->require("multiLevelRMCRTMethod",  d_multiLevelRMCRTMethod);
    //__________________________________
    // read in the regions that user would like 
    // refined if the grid has not been setup manually
    bool manualGrid;
    rmcrt_ps->getWithDefault("manualGrid", manualGrid, false);

    if(!manualGrid){
      ProblemSpecP refine_ps = rmcrt_ps->findBlock("Refine_Regions");
      if(!refine_ps ){
        string warn;
        warn ="\n INPUT FILE ERROR:\n <Refine_Regions> "
             " block not found inside of <RMCRT> block \n";
        throw ProblemSetupException(warn, __FILE__, __LINE__);
      }

      // Read in the refined regions geometry objects
      int piece_num = 0;
      list<GeometryObject::DataItem> geom_obj_data;
      geom_obj_data.push_back(GeometryObject::DataItem("level", GeometryObject::Integer));

      for (ProblemSpecP geom_obj_ps = refine_ps->findBlock("geom_object");
            geom_obj_ps != 0;
            geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

          vector<GeometryPieceP> pieces;
          GeometryPieceFactory::create(geom_obj_ps, pieces);

          GeometryPieceP mainpiece;
          if(pieces.size() == 0){
             throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
          } else if(pieces.size() > 1){
             mainpiece = scinew UnionGeometryPiece(pieces);
          } else {
             mainpiece = pieces[0];
          }
          piece_num++;
          d_refine_geom_objs.push_back(scinew GeometryObject(mainpiece,geom_obj_ps,geom_obj_data));
       }
     }

    //__________________________________
    //  bulletproofing
    if(!d_sharedState->isLockstepAMR()){
      ostringstream msg;
      msg << "\n ERROR: You must add \n"
          << " <useLockStep> true </useLockStep> \n"
          << " inside of the <AMR> section. \n"; 
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }
}
  
//______________________________________________________________________
void RMCRT_Test::scheduleInitialize ( const LevelP& level, 
                                      SchedulerP& sched )
{
  printSchedule(level,dbg,"RMCRT_Test::scheduleInitialize");

  Task* task = scinew Task( "RMCRT_Test::initialize", this, 
                            &RMCRT_Test::initialize );

  task->computes( d_colorLabel );
  task->computes( d_abskgLabel );
  sched->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
}

//______________________________________________________________________
void RMCRT_Test::scheduleComputeStableTimestep ( const LevelP& level, SchedulerP& scheduler )
{
  printSchedule(level,dbg,"RMCRT_Test::scheduleComputeStableTimestep");

  Task* task = scinew Task( "RMCRT_Test::computeStableTimestep", this, 
                            &RMCRT_Test::computeStableTimestep );

  task->computes( d_sharedState->get_delt_label(),level.get_rep() );

  scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
}
//______________________________________________________________________
void RMCRT_Test::scheduleTimeAdvance ( const LevelP& level,
                                       SchedulerP& sched)
{
  if(level->getIndex() > 0){  // only schedule once
    return;
  }

  const MaterialSet* matls = d_sharedState->allMaterials();
  GridP grid = level->getGrid();
  int maxLevels = level->getGrid()->numLevels();
  
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if(d_multiLevelRMCRTMethod){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();

    // compute Radiative properties on the finest level
    int time_sub_step = 0;
    d_RMCRT->sched_initProperties( fineLevel, sched, time_sub_step );

    // coarsen data to the coarser levels.  
    // do it in reverse order
    Task::WhichDW temp_dw = Task::NewDW;
    
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      d_RMCRT->sched_CoarsenAll (level, sched);
      d_RMCRT->sched_setBoundaryConditions( level, sched, temp_dw );
    }
    
    //__________________________________
    //  compute the extents of the rmcrt region of interest
    //  on the finest level
    d_RMCRT->sched_ROI_Extents( fineLevel, sched );
    
    // only schedule RMCRT and pseudoCFD on the finest level
    Task::WhichDW abskg_dw   = Task::NewDW;
    Task::WhichDW sigmaT4_dw = Task::NewDW;
    bool modifies_divQ       = false;
    d_RMCRT->sched_rayTrace_dataOnion(fineLevel, sched, abskg_dw, sigmaT4_dw, modifies_divQ);
    
    schedulePseudoCFD(  sched, finestPatches, matls );
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if(d_CoarseLevelRMCRTMethod){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();
   
    // compute Radiative properties on the finest level
    int time_sub_step = 0;
    d_RMCRT->sched_initProperties( fineLevel, sched, time_sub_step );
    
    
    for (int l = 0; l <= maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      
      d_RMCRT->sched_CoarsenAll (level, sched);
      
      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw   = Task::NewDW;
        Task::WhichDW sigmaT4_dw = Task::NewDW;
        bool modifies_divQ       = false;
        d_RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, modifies_divQ);
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_RMCRT->sched_Refine_Q (sched,  patches, matls);
    }

    // only schedule CFD on the finest level
    schedulePseudoCFD( sched, finestPatches, matls );
  }
}
//______________________________________________________________________
//
void RMCRT_Test::schedulePseudoCFD(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls)
{
  printSchedule(patches,dbg,"RMCRT_Test::schedulePseudoCFD");
  
  Task* t = scinew Task("RMCRT_Test::pseudoCFD",
                  this, &RMCRT_Test::pseudoCFD);
  t->requires(Task::NewDW, d_divQLabel,   d_gn, 0);
  t->requires(Task::OldDW, d_colorLabel,  d_gn, 0);
  
  t->computes( d_colorLabel );

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
void RMCRT_Test::pseudoCFD ( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw )
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"pseudoCFD");

    for(int m = 0;m<matls->size();m++){
      CCVariable<double> color;
      constCCVariable<double> divQ;
      constCCVariable<double> color_old;
      
      new_dw->allocateAndPut(color, d_colorLabel, d_matl, patch);              
      new_dw->get(divQ,       d_divQLabel,  d_matl, patch, d_gn,0);      
      old_dw->get(color_old,  d_colorLabel, d_matl, patch, d_gn,0);       
      
      color.initialize(0.0);
      
      const double rhoSTP = 118.39; // density at standard temp and pressure [g/m^3] 
      double deltaTime = 0.0243902; // !! Should be dynamic
      const double specHeat = 1.012; // specific heat [J/K]
      double rho = 0;
      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c(*iter);
        // rho = rhoSTP *298 / 1500; // more accurate, but "hard codes" temperature to 1500 
        rho = rhoSTP * 298 / color[c]; // calculate density based on ideal gas law
        color[c] = color_old[c] - (1/rho) * (1/specHeat) * deltaTime * divQ[c];
      }
      
      // set boundary conditions 
      d_RMCRT->setBC(color,  d_colorLabel->getName(), patch, d_matl);
    }
  }
}
 


//______________________________________________________________________
// STUB
void RMCRT_Test::scheduleInitialErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
{
}
//______________________________________________________________________
// STUB 
void RMCRT_Test::scheduleCoarsen ( const LevelP& coarseLevel, SchedulerP& scheduler )
{
}
//______________________________________________________________________
// STUB
void RMCRT_Test::scheduleRefine ( const PatchSet* patches, 
                                  SchedulerP& scheduler )
{
}
//______________________________________________________________________
// STUB
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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"Doing initialize");
    
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> color;
      CCVariable<double> abskg;
      new_dw->allocateAndPut(color, d_colorLabel, matl, patch);
      new_dw->allocateAndPut(abskg, d_abskgLabel, matl, patch);

      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
         IntVector idx(*iter);
         color[idx] = d_initColor;
         abskg[idx] = d_initAbskg;
       }
        // set boundary conditions 
       d_RMCRT->setBC(color,  d_colorLabel->getName(), patch, matl);
       d_RMCRT->setBC(abskg,  d_abskgLabel->getName(), patch, matl);
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


} // namespace Uintah
