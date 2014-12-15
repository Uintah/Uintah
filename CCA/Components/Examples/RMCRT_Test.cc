/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 */
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Components/Examples/RMCRT_Test.h>
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>
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
#include <Core/Grid/DbgOutput.h>
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

using namespace std;
using SCIRun::Point;
using SCIRun::Vector;

static SCIRun::DebugStream dbg("RMCRT_Test", false);


namespace Uintah
{
//______________________________________________________________________
//
RMCRT_Test::RMCRT_Test ( const ProcessorGroup* myworld ): UintahParallelComponent( myworld )
{
//  d_colorLabel    = VarLabel::create( "color",    CCVariable<double>::getTypeDescription() );
  d_divQLabel     = VarLabel::create( "divQ",     CCVariable<double>::getTypeDescription() );
//  d_abskgLabel    = VarLabel::create( "abskg",    CCVariable<double>::getTypeDescription() );
  d_cellTypeLabel = VarLabel::create( "cellType", CCVariable<int>::getTypeDescription() );

/*`==========TESTING==========*/
    #ifdef USINGFLOATRMCRT
      proc0cout << "__________________________________ USING FLOAT VERSION OF RMCRT" << endl;
      d_colorLabel  = VarLabel::create( "color",    CCVariable<float>::getTypeDescription() );
      d_abskgLabel    = VarLabel::create( "abskg",    CCVariable<float>::getTypeDescription() );
    #else
      proc0cout << "__________________________________ USING DOUBLE VERSION OF RMCRT" << endl;
      d_colorLabel    = VarLabel::create( "color",    CCVariable<double>::getTypeDescription() );
      d_abskgLabel    = VarLabel::create( "abskg",    CCVariable<double>::getTypeDescription() );
    #endif
/*===========TESTING==========`*/

  d_gac = Ghost::AroundCells;
  d_gn  = Ghost::None;
  d_matl = 0;
  d_initColor = -9;
  d_initAbskg = -9;
  d_whichAlgo = singleLevel;
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
  VarLabel::destroy(d_cellTypeLabel);

  if( d_old_uda){
    delete d_old_uda;
  }

  dbg << UintahParallelComponent::d_myworld->myrank() << " Doing: RMCRT destructor " << endl;

}

//______________________________________________________________________
//
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
  // Read in component specific variables
  ProblemSpecP me = prob_spec;
  me->getWithDefault( "calc_frequency",  d_radCalc_freq, 1 );
  me->getWithDefault( "benchmark" ,      d_benchmark,  0 );

  me->require("Temperature",  d_initColor);
  me->require("abskg",        d_initAbskg);

  // bulletproofing
  if ( d_benchmark > 5 || d_benchmark < 0  ){
     ostringstream warn;
     warn << "ERROR:  Benchmark value ("<< d_benchmark <<") not set correctly." << endl;
     warn << "Specify a value of 1 through 5 to run a benchmark case, or 0 otherwise." << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  RMCRT variables
  if (prob_spec->findBlock("RMCRT")){
    ProblemSpecP rmcrt_ps = prob_spec->findBlock("RMCRT");

/*`==========TESTING==========*/
    #ifdef USINGFLOATRMCRT
      proc0cout << "__________________________________ USING FLOAT VERSION OF RMCRT" << endl;
      d_RMCRT = scinew floatRay();
    #else
      proc0cout << "__________________________________ USING DOUBLE VERSION OF RMCRT" << endl;
      d_RMCRT = scinew Ray();
    #endif
/*===========TESTING==========`*/

    d_RMCRT->registerVarLabels(0,d_abskgLabel,
                                 d_colorLabel,
                                 d_cellTypeLabel,
                                 d_divQLabel);
    proc0cout << "__________________________________ Reading in RMCRT section of ups file" << endl;
    d_RMCRT->problemSetup( prob_spec, rmcrt_ps, grid, d_sharedState );

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
      } else if ( type == "RMCRT_coarseLevel" ) {   // 2 Level
        d_whichAlgo = coarseLevel;
      } else if ( type == "singleLevel" ) {         // 1 LEVEL
        d_whichAlgo = singleLevel;
      } else if ( type == "radiometerOnly" ) {      // Only when radiometer is used
        d_whichAlgo = radiometerOnly;
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


    bool foundIndex = false;
    int timeIndex = -9;
    for (unsigned int i = 0; i < index.size(); i++) {
      if( (int) d_old_uda->timestep == index[i] ){
        foundIndex = true;
        timeIndex = i;
      }
    }

    if( ! foundIndex ){
      ostringstream warn;
      warn << "The timestep ("<< d_old_uda->timestep << ") was not found in the uda\n"
           << "There are " << index.size() << " timesteps\n";
      throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
    }

    // are the grids the same ?
    GridP uda_grid = archive->queryGrid( timeIndex );
    areGridsEqual(uda_grid.get_rep(), grid.get_rep());


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
//
//______________________________________________________________________
void RMCRT_Test::scheduleInitialize ( const LevelP& level,
                                      SchedulerP& sched )
{


  Task* task = NULL;
  if (!d_old_uda) {
    task = scinew Task( "RMCRT_Test::initialize", this,
                        &RMCRT_Test::initialize );
  } else {
    task = scinew Task( "RMCRT_Test::initializeWithUda", this,
                        &RMCRT_Test::initializeWithUda );
  }

  printSchedule(level,dbg,"RMCRT_Test::scheduleInitialize");
  task->computes( d_colorLabel );
  task->computes( d_abskgLabel );
  task->computes( d_cellTypeLabel );
  sched->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
}

//______________________________________________________________________
//
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
//
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
  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    d_RMCRT->sched_CarryForward_Var( level, sched, d_cellTypeLabel );
    d_RMCRT->sched_CarryForward_Var( level, sched, d_colorLabel );
    d_RMCRT->sched_CarryForward_Var( level, sched, d_abskgLabel );
  }

  Radiometer* radiometer = d_RMCRT->getRadiometer();

  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if( d_whichAlgo == dataOnion ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    Task::WhichDW temp_dw = Task::NewDW;

    // modify Radiative properties on the finest level
    sched_initProperties( fineLevel, sched, d_radCalc_freq);

    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, d_radCalc_freq, false );

    d_RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw, d_radCalc_freq );

    // coarsen data to the coarser levels.
    // do it in reverse order
    Task::WhichDW notUsed = Task::OldDW;
    const bool backoutTemp = true;

    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = true;
      const bool modifies_sigmaT4 = false;
      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, d_radCalc_freq );
      d_RMCRT->sched_setBoundaryConditions( level, sched, notUsed, d_radCalc_freq, backoutTemp );
    }

    //__________________________________
    //  compute the extents of the rmcrt region of interest
    //  on the finest level
    d_RMCRT->sched_ROI_Extents( fineLevel, sched );

    Task::WhichDW abskg_dw   = Task::NewDW;
    Task::WhichDW sigmaT4_dw = Task::NewDW;
    const bool modifies_divQ = false;
    d_RMCRT->sched_rayTrace_dataOnion(fineLevel, sched, abskg_dw, sigmaT4_dw, modifies_divQ, d_radCalc_freq);

  }

  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if( d_whichAlgo == coarseLevel ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    Task::WhichDW temp_dw = Task::NewDW;

    // modify Radiative properties on the finest level
    sched_initProperties( fineLevel, sched, d_radCalc_freq );

    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, d_radCalc_freq, false );

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = true;
      const bool modifies_sigmaT4 = false;

      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, d_radCalc_freq);

      if(level->hasFinerLevel() || maxLevels == 1){
        Task::WhichDW abskg_dw    = Task::NewDW;
        Task::WhichDW sigmaT4_dw  = Task::NewDW;
        Task::WhichDW celltype_dw = Task::NewDW;
        const bool modifies_divQ  = false;
        const bool backoutTemp    = true;

        d_RMCRT->sched_setBoundaryConditions( level, sched, temp_dw, d_radCalc_freq, backoutTemp );
        
        if (radiometer ){
          radiometer->sched_initializeRadVars( level, sched, d_radCalc_freq );
        }
        
        d_RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, d_radCalc_freq );
      }
    }

    //__________________________________
    // interpolate divQ on coarse level -> fine level
    for (int l = 1; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_RMCRT->sched_Refine_Q (sched,  patches, matls, d_radCalc_freq);
    }
  }
  
  //______________________________________________________________________
  //   1 - L E V E L   A P P R O A C H
  //  RMCRT is performed on one level
  if( d_whichAlgo == singleLevel ){
    Task::WhichDW temp_dw = Task::NewDW;
    
    sched_initProperties( level, sched, d_radCalc_freq );

    d_RMCRT->sched_sigmaT4( level,  sched, temp_dw, d_radCalc_freq, false );
    
    Task::WhichDW abskg_dw    = Task::NewDW;                                                                       
    Task::WhichDW sigmaT4_dw  = Task::NewDW;                                                                       
    Task::WhichDW celltype_dw = Task::NewDW;
    const bool modifies_divQ  = false;
    const bool backoutTemp    = true;                                                                   
    
    d_RMCRT->sched_setBoundaryConditions( level, sched, temp_dw, d_radCalc_freq, backoutTemp );
    
    if (radiometer ){
      radiometer->sched_initializeRadVars( level, sched, d_radCalc_freq );
    }
    
    d_RMCRT->sched_rayTrace(level, sched, abskg_dw, sigmaT4_dw, celltype_dw, modifies_divQ, d_radCalc_freq ); 
  }
  
  //______________________________________________________________________
  //   R A D I O M E T E R  
  //  No other calculations 
  if( d_whichAlgo == radiometerOnly ){
    
    Task::WhichDW temp_dw     = Task::NewDW;
    Task::WhichDW abskg_dw    = Task::NewDW;
    Task::WhichDW sigmaT4_dw  = Task::NewDW;
    Task::WhichDW celltype_dw = Task::NewDW;
    const bool includeEC      = true;
    const bool backoutTemp    = true;

    sched_initProperties( level, sched, d_radCalc_freq );
    
    radiometer->sched_sigmaT4( level, sched, temp_dw, d_radCalc_freq, includeEC );
    
    d_RMCRT->sched_setBoundaryConditions( level, sched, temp_dw, d_radCalc_freq, backoutTemp );
    
    radiometer->sched_initializeRadVars( level, sched, d_radCalc_freq );

    radiometer->sched_radiometer( level, sched, abskg_dw, sigmaT4_dw, celltype_dw, d_radCalc_freq );

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
//
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

/*`==========TESTING==========*/
    #ifdef USINGFLOATRMCRT
      CCVariable<float> color;
      CCVariable<float> abskg;
      bool isFloat = true;
    #else
      CCVariable<double> color;
      CCVariable<double> abskg;
      bool isFloat = false;
    #endif
/*===========TESTING==========`*/

      CCVariable<int> cellType;
      new_dw->allocateAndPut(color,    d_colorLabel,    matl, patch);
      new_dw->allocateAndPut(abskg,    d_abskgLabel,    matl, patch);
      new_dw->allocateAndPut(cellType, d_cellTypeLabel, matl, patch);

      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
         IntVector idx(*iter);
         color[idx] = d_initColor;
         abskg[idx] = d_initAbskg;
      }


#ifdef USINGFLOATRMCRT
      //__________________________________
      // HACK:
      //  Make a copy of the float data and place it
      //  in a double array.  The BC infrastructure can't
      //  deal with floats
      CCVariable<double> D_abskg, D_color;

      new_dw->allocateTemporary(D_abskg, patch);
      new_dw->allocateTemporary(D_color, patch);
      for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        D_abskg[c] = (double) abskg[c];
        D_color[c] = (double) color[c];
      }
      // set boundary conditions
      d_RMCRT->setBC<double>( D_color,  d_colorLabel->getName(), patch, matl );
      d_RMCRT->setBC<double>( D_abskg,  d_abskgLabel->getName(), patch, matl );

      for (CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        abskg[c] = (float) D_abskg[c];
        color[c]  = (float) D_color[c];
      }
#else
      d_RMCRT->setBC<double>( color,  d_colorLabel->getName(), patch, matl );
      d_RMCRT->setBC<double>( abskg,  d_abskgLabel->getName(), patch, matl );

#endif

      //__________________________________
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
// Execute this tasks on all levels even though the Data-onion only needs
// needs data from the finest levels.
//______________________________________________________________________
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
  int timeIndex = -9;

  for (unsigned int i = 0; i < index.size(); i++) {
    if( (int) d_old_uda->timestep == index[i] ){
      timeIndex = i;
    }
  }

  GridP uda_grid = archive->queryGrid(timeIndex);

  const Level*  uda_level = uda_grid->getLevel(0).get_rep();        // there's only one level in these problem
  const int uda_matl = d_old_uda->matl;

  //__________________________________
  //  Cell Type: loop over the UDA patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    IntVector low  = patch->getExtraCellLowIndex();
    IntVector high = patch->getExtraCellHighIndex();

    CCVariable<int> cellType, uda_cellType;
    new_dw->allocateAndPut(cellType, d_cellTypeLabel, d_matl, patch);
    cellType.initialize(d_flow_cell);

    if (d_old_uda->cellTypeName != "NONE" ){
      archive->queryRegion( uda_cellType,  d_old_uda->cellTypeName, uda_matl, uda_level, timeIndex, low, high);
      cellType.copyData(uda_cellType);
    }
  }

  //__________________________________
  // abskg and temperature
  // Note the user may have saved the data as a float so you
  // must take that into account.
  // Determine what type (float/double) the variables were saved as
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  const Uintah::TypeDescription* subType = NULL;

  archive->queryVariables(vars, types);

  for (unsigned int i = 0; i < vars.size(); i++) {
    if (d_old_uda->abskgName == vars[i]) {
      subType = types[i]->getSubType();
    } else if (d_old_uda->temperatureName == vars[i]){
      subType = types[i]->getSubType();
    }
  }

  //__________________________________
  //  Load abskg & temperature from old uda into new data warehouse
  proc0cout << "Extracting data from " << d_old_uda->udaName
          << " at time " << times[timeIndex]
          << " and initializing RMCRT variables " << endl;

  // loop over the UDA patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"Doing initializeWithUda");

/*`==========TESTING==========*/
    #ifdef USINGFLOATRMCRT
      CCVariable<float> color;
      CCVariable<float> abskg;
    #else
      CCVariable<double> color;
      CCVariable<double> abskg;
    #endif
/*===========TESTING==========`*/

    new_dw->allocateAndPut(color,    d_colorLabel,    d_matl, patch);
    new_dw->allocateAndPut(abskg,    d_abskgLabel,    d_matl, patch);

    IntVector low  = patch->getExtraCellLowIndex();
    IntVector high = patch->getExtraCellHighIndex();

    //             D O U B L E
    if ( subType->getType() == Uintah::TypeDescription::double_type ) {
      CCVariable<double> uda_temp;
      CCVariable<double> uda_abskg;

      archive->queryRegion( uda_temp,  d_old_uda->temperatureName, uda_matl, uda_level, timeIndex, low, high);
      archive->queryRegion( uda_abskg, d_old_uda->abskgName,       uda_matl, uda_level, timeIndex, low, high);

      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c(*iter);
        color[c] = uda_temp[c];
        abskg[c] = uda_abskg[c];
      }
    //            F L O A T
    }else if( subType->getType() == Uintah::TypeDescription::float_type ) {
      CCVariable<float> uda_temp;
      CCVariable<float> uda_abskg;

      archive->queryRegion( uda_temp,  d_old_uda->temperatureName, uda_matl, uda_level, timeIndex, low, high);
      archive->queryRegion( uda_abskg, d_old_uda->abskgName,       uda_matl, uda_level, timeIndex, low, high);

      for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c(*iter);
        color[c] = uda_temp[c];
        abskg[c] = uda_abskg[c];
      }
    }

    d_RMCRT->setBC( color,  d_colorLabel->getName(), patch, d_matl );
    d_RMCRT->setBC( abskg,  d_abskgLabel->getName(), patch, d_matl );
  }
  delete archive;
}
//______________________________________________________________________
//
//______________________________________________________________________
void RMCRT_Test::sched_initProperties( const LevelP& level,
                                        SchedulerP& sched,
                                        const int radCalc_freq )
{

  if( d_benchmark != 0 ){
    Task* tsk = scinew Task( "RMCRT_Test::initProperties", this,
                             &RMCRT_Test::initProperties, radCalc_freq);

    printSchedule(level,dbg,"RMCRT_Test::initProperties");

    tsk->modifies( d_colorLabel );
    tsk->modifies( d_abskgLabel );
    tsk->modifies( d_cellTypeLabel );

    sched->addTask( tsk, level->eachPatch(), d_sharedState->allMaterials() );
  }
}
//______________________________________________________________________
//  Initialize the properties
//______________________________________________________________________
void RMCRT_Test::initProperties( const ProcessorGroup* pc,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const int radCalc_freq )
{
  // Only run if it's time
  if ( d_RMCRT->doCarryForward( radCalc_freq ) ) {
    return;
  }

  const Level* level = getLevel(patches);

  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    printTask(patches,patch,dbg,"Doing RMCRT_test::InitProperties");

  /*`==========TESTING==========*/
    #ifdef USINGFLOATRMCRT
      CCVariable<float> color;
      CCVariable<float> abskg;
    #else
      CCVariable<double> color;
      CCVariable<double> abskg;
    #endif
/*===========TESTING==========`*/

    new_dw->getModifiable( abskg,    d_abskgLabel,     d_matl, patch );
    abskg.initialize  ( 0.0 );

    IntVector pLow;
    IntVector pHigh;
    level->findInteriorCellIndexRange(pLow, pHigh);

    int Nx = pHigh[0] - pLow[0];
    int Ny = pHigh[1] - pLow[1];
    int Nz = pHigh[2] - pLow[2];

    Vector Dx = patch->dCell();

    BBox L_BB;
    level->getInteriorSpatialRange(L_BB);                 // edge of computational domain
    Vector L_length = Abs(L_BB.max() - L_BB.min());

    //__________________________________
    //  Benchmark initializations
    if ( d_benchmark == 1 || d_benchmark == 3 ) {

      // bulletproofing
      Vector valid_length(1,1,1);
      if (L_length != valid_length){
        ostringstream msg;
        msg << "\n RMCRT:ERROR: the benchmark problem selected is only valid on the domain \n";
        msg << valid_length << ".  Your domain is " << L_BB << endl;
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx - 1.0) /2.0) * Dx[0]) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Ny - 1.0) /2.0) * Dx[1]) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nz - 1.0) /2.0) * Dx[2]) )
                        + 0.1;
      }
    }

    if ( d_benchmark == 2 ) {              // This section should be simplified/ cleaned up --Todd

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 1;
      }
    }

    if( d_benchmark == 3) {
      CCVariable<double> temp;
      new_dw->getModifiable(temp, d_colorLabel, d_matl, patch);

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        temp[c] = 1000 * abskg[c];

      }
    }

    if( d_benchmark == 4 ) {  // Siegel isotropic scattering
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = d_initAbskg;

      }
    }

    if( d_benchmark == 5 ) {  // Siegel isotropic scattering for specific abskg and sigma_scat
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 2;
        //d_sigmaScat = 8;
      }
    }
  }
}


//______________________________________________________________________
//
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
//
//  This is basically the == operator from Grid.h but slightly tweaked
//______________________________________________________________________
void RMCRT_Test::areGridsEqual( const GridP& uda_grid,
                                const GridP& grid)
{
  bool areEqual = true;

  int nLevels = grid->numLevels()-1;
  const LevelP level      = grid->getLevel(nLevels);      // finest grid is on the last level
  const LevelP otherlevel = uda_grid->getLevel(0);        // there's only one level in these problem
  if (level->numPatches() != otherlevel->numPatches())
    areEqual = false;

  // do the patches have the same number of cells and
  // cover the same physical domain?
  Level::const_patchIterator iter      = level->patchesBegin();
  Level::const_patchIterator otheriter = otherlevel->patchesBegin();
  for (; iter != level->patchesEnd(); iter++, otheriter++) {
    const Patch* patch = *iter;
    const Patch* otherpatch = *otheriter;

    IntVector lo, o_lo;
    IntVector hi, o_hi;
    lo   = patch->getCellLowIndex();
    hi   = patch->getCellHighIndex();
    o_lo = otherpatch->getCellLowIndex();
    o_hi = otherpatch->getCellHighIndex();

    if ( lo !=  o_lo || hi != o_hi ){
      areEqual = false;
    }
    if( patch->getCellPosition(lo) != otherpatch->getCellPosition(o_lo) ||
        patch->getCellPosition(hi) != otherpatch->getCellPosition(o_hi) ){
      areEqual = false;
    }
  }

  if ( !areEqual ) {
    ostringstream warn;
    warn << "\n\nERROR initalizeUsingUda: The grid defined in the input file"
         <<  " is not equal to the initialization uda's grid\n"
         <<  " For the Data Onion algorithm the finest level in the input file must be equal";
    throw ProblemSetupException(warn.str(),__FILE__,__LINE__);
  }
}

} // namespace Uintah
