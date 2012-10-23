/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Components/Examples/RMCRT_Test.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/BBox.h>
#include <Core/Grid/Box.h>
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
  d_cellTypeLabel = VarLabel::create( "cellType", CCVariable<int>::getTypeDescription() );
   
  d_gac = Ghost::AroundCells;
  d_gn  = Ghost::None;
  d_matl = 0;
  d_initColor = -9;
  d_initAbskg = -9;
  d_whichAlgo = coarseLevel;
  d_wall_cell = 8; //<----HARD CODED WALL CELL
  d_flow_cell = -1; //<----HARD CODED FLOW CELL 
  
  d_old_uda = 0; 
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
  VarLabel::destroy(d_cellTypeLabel); 
  
  if( d_old_uda){
    delete d_old_uda;
  }
  
  dbg << UintahParallelComponent::d_myworld->myrank() << " Doing: RMCRT destructor " << endl;

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
  sched->overrideVariableBehavior("color",false, false, true, false, false);

  //__________________________________
  //  RMCRT
  if (prob_spec->findBlock("RMCRT")){
    ProblemSpecP rmcrt_ps = prob_spec->findBlock("RMCRT"); 
    
#ifdef HAVE_CUDA
    d_RMCRT = scinew Ray(dynamic_cast<UnifiedScheduler*>(getPort("scheduler")));
#else
    d_RMCRT = scinew Ray();
#endif

    d_RMCRT->registerVarLabels(0,d_abskgLabel,
                                 d_absorpLabel,
                                 d_colorLabel,
                                 d_cellTypeLabel, 
                                 d_divQLabel);
                                 
    rmcrt_ps->require("Temperature",  d_initColor);
    rmcrt_ps->require("abskg",        d_initAbskg);
    
    d_RMCRT->problemSetup( prob_spec, rmcrt_ps );
    proc0cout << "__________________________________ Reading in RMCRT section of ups file" << endl;
    
      
    //__________________________________
    //  Read in the dataOnion section
    ProblemSpecP alg_ps = rmcrt_ps->findBlock("algorithm");
    if (alg_ps){
    
      string type="NULL";
      alg_ps->getAttribute("type", type);
    
      if (type == "dataOnion" ) {
        d_whichAlgo = dataOnion;

        //__________________________________
        //  bulletproofing
        if(!d_sharedState->isLockstepAMR()){
          ostringstream msg;
          msg << "\n ERROR: You must add \n"
              << " <useLockStep> true </useLockStep> \n"
              << " inside of the <AMR> section. \n"; 
          throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
        }
      } else if ( type == "RMCRT_coarseLevel" ) {
        d_whichAlgo = coarseLevel;
      }
    }

  }
  
  //__________________________________
  //  Read in initalizeUsingUda section
  ProblemSpecP uda_ps = prob_spec->findBlock("initalizeUsingUda");
  if(uda_ps){
    d_old_uda = scinew useOldUdaData();
    uda_ps->get( "uda_name"  ,          d_old_uda->udaName );
    uda_ps->get( "timestep"  ,          d_old_uda->timestep );
    uda_ps->get( "abskg_varName"  ,     d_old_uda->abskgName );
    uda_ps->get( "temperature_varName", d_old_uda->temperatureName );
    uda_ps->getWithDefault( "matl_index",          d_old_uda->matl, 0 );
    uda_ps->getWithDefault( "cellType_varName"  ,  d_old_uda->cellTypeName, "NONE" );
  }
  
  //__________________________________
  //  Intrusions
  if (prob_spec->findBlock("Intrusion")){
    ProblemSpecP intrusion_ps = prob_spec->findBlock("Intrusion"); 
    ProblemSpecP geom_obj_ps = intrusion_ps->findBlock("geom_object");
    GeometryPieceFactory::create(geom_obj_ps, d_intrusion_geom);
  }
  
  
  //__________________________________
  //  General bullet proofing
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
  // usingOldUda bullet proofing
  if(d_old_uda){    
    DataArchive* archive = scinew DataArchive(d_old_uda->udaName);
    
    // does the user specified timestep exist?
    vector<int> index;
    vector<double> times;
    archive->queryTimesteps(index, times);
  
    if( d_old_uda->timestep >= index.size() ){
      ostringstream warn;
      warn << "The timestep ("<< d_old_uda->timestep << ") was not found in the uda\n"
           << "There are " << index.size()-1 << " timesteps\n";
      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
    }   
    
    // are the grids the same ?
    GridP uda_grid = archive->queryGrid(d_old_uda->timestep); 
    
    if( ( *(uda_grid.get_rep() )  == *( grid.get_rep() ) ) == false ){
      ostringstream warn;
      warn << "\n\nERROR initalizeUsingUda: The grid defined in the input file"
           <<  " is not equal to the initialization uda's grid\n";
      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);  
    }
    
    // do the variables exist
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    int vars_found = 0;
    
    for (unsigned int i = 0; i < vars.size(); i++) {
      if (d_old_uda->abskgName == vars[i]) {
        vars_found +=1;
      } else if (d_old_uda->temperatureName == vars[i]){
        vars_found +=1;
      }
    }
   
    if (vars_found != 2 && vars_found != 3) {
      ostringstream warn;
      warn << "The variables (" << d_old_uda->abskgName << "),"
           << " (" << d_old_uda->temperatureName << "), or"
           << " optional variale cellType: (" << d_old_uda->cellTypeName 
           << ") was not found in the uda";
      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);        
    }
    delete archive;
  }
}
  
//______________________________________________________________________
void RMCRT_Test::scheduleInitialize ( const LevelP& level, 
                                      SchedulerP& sched )
{
  printSchedule(level,dbg,"RMCRT_Test::scheduleInitialize");

  Task* task = NULL;
  if (!d_old_uda) {
    task = scinew Task( "RMCRT_Test::initialize", this, 
                        &RMCRT_Test::initialize );  
  } else {
    task = scinew Task( "RMCRT_Test::initializeWithUda", this, 
                        &RMCRT_Test::initializeWithUda );
  }

  task->computes( d_colorLabel );
  task->computes( d_abskgLabel );
  task->computes( d_cellTypeLabel ); 
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
  
  // move data to the new_dw for simplicity
  for (int l = 0; l <= maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    d_RMCRT->sched_CarryForward (level, sched, d_cellTypeLabel);
    d_RMCRT->sched_CarryForward (level, sched, d_colorLabel);
    d_RMCRT->sched_CarryForward (level, sched, d_abskgLabel);
  }
  
  
  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if( d_whichAlgo == dataOnion ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();
    Task::WhichDW temp_dw = Task::NewDW;
    
    // modify Radiative properties on the finest level
    d_RMCRT->sched_initProperties( fineLevel, sched);
    
    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, false );
 
    d_RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw );
        
    // coarsen data to the coarser levels.  
    // do it in reverse order
    Task::WhichDW notUsed = Task::OldDW;
    const bool backoutTemp = true;
    
    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = true;
      const bool modifies_sigmaT4 = false;
      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4);
      d_RMCRT->sched_setBoundaryConditions( level, sched, notUsed, backoutTemp );
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
    
//    schedulePseudoCFD(  sched, finestPatches, matls );
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if( d_whichAlgo == coarseLevel ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();
    Task::WhichDW temp_dw = Task::NewDW;
   
    // modify Radiative properties on the finest level
    d_RMCRT->sched_initProperties( fineLevel, sched  );
    
    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, false );
    
    d_RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw );
    
    
    for (int l = 0; l <= maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = true;
      const bool modifies_sigmaT4 = false;
      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4);
      
      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw    = Task::NewDW;
        Task::WhichDW sigmaT4_dw  = Task::NewDW;
        Task::WhichDW celltype_dw = Task::NewDW;
        bool modifies_divQ       = false;
        d_RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ);
      }
    }

    // push divQ  to the coarser levels 
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_RMCRT->sched_Refine_Q (sched,  patches, matls);
    }

    // only schedule CFD on the finest level
//    schedulePseudoCFD( sched, finestPatches, matls );
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
                  
  t->requires(Task::NewDW, d_divQLabel,  d_gn, 0);
  t->modifies( d_colorLabel );

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
                 
      new_dw->get(           divQ,     d_divQLabel,  d_matl, patch, d_gn,0);
      new_dw->getModifiable( color,    d_colorLabel, d_matl, patch );
      
      
      const double rhoSTP = 118.39; // density at standard temp and pressure [g/m^3] 
      double deltaTime = 0.0243902; // !! Should be dynamic
      const double specHeat = 1.012; // specific heat [J/K]
      double rho = 0;
      
      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c(*iter);
        // rho = rhoSTP *298 / 1500; // more accurate, but "hard codes" temperature to 1500 
        //__________________________________
        //
        double color_wrong = 0.0;            // THIS IS WRONG
        //__________________________________

        rho = rhoSTP * 298 / color_wrong; // calculate density based on ideal gas law
        
        color[c] = color[c] - (1/rho) * (1/specHeat) * deltaTime * divQ[c];
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

    Box patch_box = patch->getBox(); 
    
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> color;
      CCVariable<double> abskg;
      CCVariable<int> cellType; 
      new_dw->allocateAndPut(color,    d_colorLabel,    matl, patch);
      new_dw->allocateAndPut(abskg,    d_abskgLabel,    matl, patch);
      new_dw->allocateAndPut(cellType, d_cellTypeLabel, matl, patch); 

      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
         IntVector idx(*iter);
         color[idx] = d_initColor;
         abskg[idx] = d_initAbskg;
      }
      // set boundary conditions 
      d_RMCRT->setBC(color,  d_colorLabel->getName(), patch, matl);
      d_RMCRT->setBC(abskg,  d_abskgLabel->getName(), patch, matl);

      // initialize cell type
      cellType.initialize(d_flow_cell);

      for ( int i = 0; i < (int)d_intrusion_geom.size(); i++ ){

        for ( std::vector<GeometryPieceP>::iterator iter=d_intrusion_geom.begin(); iter!=d_intrusion_geom.end(); iter++ ){
                  
          GeometryPieceP piece = iter[i]; 
          Box geometry_box     = piece->getBoundingBox(); 
          Box intersect_box    = geometry_box.intersect( patch_box ); 
        
          if ( !(intersect_box.degenerate()) ) { 

            for ( CellIterator cell_iter(patch->getExtraCellIterator()); !cell_iter.done(); cell_iter++) {
              IntVector c = *cell_iter; 
              Point p = patch->cellPosition( c ); 
              if ( piece->inside( p ) ) {
                
                cellType[c] = d_wall_cell; 
               
              }
            }
          }
        }
      }
    }
  }
}

//______________________________________________________________________
// initialize using data from a previously run uda.
void RMCRT_Test::initializeWithUda (const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{
  //__________________________________
  // retreive data from uda
  DataArchive* archive = scinew DataArchive( d_old_uda->udaName );
  vector<int> index;
  vector<double> times;
  archive->queryTimesteps(index, times);
  GridP uda_grid = archive->queryGrid(d_old_uda->timestep);

  const int timestep = d_old_uda->timestep;
  const int uda_matl = d_old_uda->matl;

  vector<CCVariable<double>*> uda_temp(     patches->size() );
  vector<CCVariable<double>*> uda_abskg(    patches->size() );
  vector<CCVariable<int>*>    uda_cellType( patches->size() );

  proc0cout << "Extracting data from " << d_old_uda->udaName
            << " at time " << times[timestep] 
            << " and initializing RMCRT variables " << endl;
            
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  

    uda_temp[p]  = scinew CCVariable<double>;
    uda_abskg[p] = scinew CCVariable<double>;
    
    archive->query( *(CCVariable<double>*) uda_temp[p],  
                    d_old_uda->temperatureName, uda_matl, patch, timestep);

    archive->query( *(CCVariable<double>*) uda_abskg[p], 
                    d_old_uda->abskgName,       uda_matl, patch, timestep);

    if (d_old_uda->cellTypeName != "NONE" ){
      uda_cellType[p] = scinew CCVariable<int>;

      archive->query( *(CCVariable<int>*) uda_cellType[p],  
                      d_old_uda->cellTypeName, uda_matl, patch, timestep);  
    }
  }
  delete archive;

  //__________________________________
  //  initialize 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"Doing initializeWithUda");    

    CCVariable<double> color;
    CCVariable<double> abskg;
    CCVariable<int> cellType; 
    new_dw->allocateAndPut(color,    d_colorLabel,    d_matl, patch);
    new_dw->allocateAndPut(abskg,    d_abskgLabel,    d_matl, patch);
    new_dw->allocateAndPut(cellType, d_cellTypeLabel, d_matl, patch); 
    cellType.initialize(d_flow_cell);

    for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
      IntVector c(*iter);                      
      color[c] = (*uda_temp[p])[c];            
      abskg[c] = (*uda_abskg[p])[c];           

      if (d_old_uda->cellTypeName != "NONE" ){ 
       cellType[c] = (*uda_cellType[p])[c];    
      }
    }

    // set boundary conditions 
    d_RMCRT->setBC(color,  d_colorLabel->getName(), patch, d_matl);
    d_RMCRT->setBC(abskg,  d_abskgLabel->getName(), patch, d_matl);
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
