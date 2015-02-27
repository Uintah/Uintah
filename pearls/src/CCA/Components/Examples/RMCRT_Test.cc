/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
  d_colorLabel    = VarLabel::create( "color",    CCVariable<double>::getTypeDescription() );
  d_divQLabel     = VarLabel::create( "divQ",     CCVariable<double>::getTypeDescription() );
  d_compAbskgLabel= VarLabel::create( "abskg",    CCVariable<double>::getTypeDescription() );
  d_cellTypeLabel = VarLabel::create( "cellType", CCVariable<int>::getTypeDescription() );

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
  //if ( d_RMCRT ) {
  //  delete d_RMCRT;
  //}

  VarLabel::destroy(d_colorLabel);
  VarLabel::destroy(d_divQLabel);
  VarLabel::destroy(d_compAbskgLabel);
  VarLabel::destroy(d_cellTypeLabel);

  if ( d_RMCRT ) {
    delete d_RMCRT;
  }

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
    
    // using float or doubles for all-to-all variables
    map<string,string> type;
    rmcrt_ps->getAttributes(type);
    
    string isFloat = type["type"];
    
    if( isFloat == "float" ){
      d_RMCRT = scinew Ray( TypeDescription::float_type );
    } else {
      d_RMCRT = scinew Ray( TypeDescription::double_type );
    }

    d_RMCRT->registerVarLabels(0,d_compAbskgLabel,
                                 d_colorLabel,
                                 d_cellTypeLabel,
                                 d_divQLabel);
    proc0cout << "__________________________________ Reading in RMCRT section of ups file" << endl;
    d_RMCRT->problemSetup( prob_spec, rmcrt_ps, grid, d_sharedState );
    
    d_RMCRT->BC_bulletproofing( rmcrt_ps );

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
  task->computes( d_compAbskgLabel );
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

  // move fields to the new_dw to mimic what a component would compute
  // The variables temperature, cellType, and abskg are computed on the finest level
  const LevelP& finestLevel = grid->getLevel( maxLevels -1 );
  sched_initProperties( finestLevel, sched, d_radCalc_freq );

  Radiometer* radiometer = d_RMCRT->getRadiometer();

  //______________________________________________________________________
  //   D A T A   O N I O N   A P P R O A C H
  if( d_whichAlgo == dataOnion ){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    Task::WhichDW temp_dw  = Task::NewDW;
    Task::WhichDW abskg_dw = Task::NewDW;
    
    // convert abskg:dbl -> abskg:flt if needed
    d_RMCRT->sched_DoubleToFloat( fineLevel,sched, abskg_dw, d_radCalc_freq );

    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, d_radCalc_freq, false );

    d_RMCRT->sched_setBoundaryConditions( fineLevel, sched, temp_dw, d_radCalc_freq );

    // coarsen data to the coarser levels.
    // do it in reverse order
    Task::WhichDW notUsed = Task::OldDW;
    const bool backoutTemp = true;

    for (int l = maxLevels - 2; l >= 0; l--) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;
      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, d_radCalc_freq );
      d_RMCRT->sched_setBoundaryConditions( level, sched, notUsed, d_radCalc_freq, backoutTemp );
    }

    //__________________________________
    //  compute the extents of the rmcrt region of interest
    //  on the finest level
    d_RMCRT->sched_ROI_Extents( fineLevel, sched );

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
    Task::WhichDW temp_dw  = Task::NewDW;
    Task::WhichDW abskg_dw = Task::NewDW;
    
    // convert abskg:dbl -> abskg:flt if needed
    d_RMCRT->sched_DoubleToFloat( fineLevel,sched, abskg_dw, d_radCalc_freq );

    d_RMCRT->sched_sigmaT4( fineLevel,  sched, temp_dw, d_radCalc_freq, false );

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const bool modifies_abskg   = false;
      const bool modifies_sigmaT4 = false;

      d_RMCRT->sched_CoarsenAll (level, sched, modifies_abskg, modifies_sigmaT4, d_radCalc_freq);

      if( level->hasFinerLevel() ){
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
    Task::WhichDW temp_dw  = Task::NewDW;
    Task::WhichDW abskg_dw = Task::NewDW;

    // convert abskg:dbl -> abskg:flt if needed
    d_RMCRT->sched_DoubleToFloat( level,sched, abskg_dw, d_radCalc_freq );

    d_RMCRT->sched_sigmaT4( level,  sched, temp_dw, d_radCalc_freq, false );
                                                                   
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
    
    // convert abskg:dbl -> abskg:flt if needed
    d_RMCRT->sched_DoubleToFloat( level, sched, abskg_dw, d_radCalc_freq );
    
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
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"Doing initialize");

    Box patch_box = patch->getBox();

    int matl = 0;
    CCVariable<double> color;
    CCVariable<double> abskg;

    CCVariable<int> cellType;
    new_dw->allocateAndPut(color,    d_colorLabel,    matl, patch);
    new_dw->allocateAndPut(abskg,    d_compAbskgLabel,matl, patch);
    new_dw->allocateAndPut(cellType, d_cellTypeLabel, matl, patch);

    //__________________________________
    //  Default initialization
    for ( CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
       IntVector idx(*iter);
       color[idx] = d_initColor;
       abskg[idx] = d_initAbskg;
    }

    //__________________________________
    //  Benchmark initializations
    if ( d_benchmark == 1 || d_benchmark == 3 ) {

      // bulletproofing
      BBox L_BB;
      level->getInteriorSpatialRange(L_BB);                 // edge of computational domain
      Vector L_length = Abs(L_BB.max() - L_BB.min());

      Vector valid_length(1,1,1);
      if (L_length != valid_length){
        ostringstream msg;
        msg << "\n RMCRT:ERROR: the benchmark problem selected is only valid on the domain \n";
        msg << valid_length << ".  Your domain is " << L_BB << endl;
        throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
      }

      IntVector pLow, pHigh;
      level->findInteriorCellIndexRange(pLow, pHigh);
      IntVector Nx = pHigh - pLow;
      Vector Dx = patch->dCell();

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = 0.90 * ( 1.0 - 2.0 * fabs( ( c[0] - (Nx.x() - 1.0) /2.0) * Dx.x() ) )
                        * ( 1.0 - 2.0 * fabs( ( c[1] - (Nx.y() - 1.0) /2.0) * Dx.y() ) )
                        * ( 1.0 - 2.0 * fabs( ( c[2] - (Nx.z() - 1.0) /2.0) * Dx.z() ) )
                        + 0.1;
      }
    }

    if ( d_benchmark == 2 ) {
      abskg.initialize( 1 );
    }

    if( d_benchmark == 3) {
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        color[c] = 1000 * abskg[c];
      }
    }

    if( d_benchmark == 4 ) {  // Siegel isotropic scattering
      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){
        IntVector c = *iter;
        abskg[c] = d_initAbskg;

      }
    }

    if( d_benchmark == 5 ) {  // Siegel isotropic scattering for specific abskg and sigma_scat
      abskg.initialize( 2.0 );
    }

    d_RMCRT->setBC<double, double>( color,  d_colorLabel->getName(),     patch, matl );
    d_RMCRT->setBC<double, double>( abskg,  d_compAbskgLabel->getName(), patch, matl );

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
    }  // intrusion
  }  // patch
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

    CCVariable<double> color;
    CCVariable<double> abskg;

    new_dw->allocateAndPut(color,    d_colorLabel,     d_matl, patch);
    new_dw->allocateAndPut(abskg,    d_compAbskgLabel, d_matl, patch);

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

    d_RMCRT->setBC<double, double>( color,  d_colorLabel->getName(),     patch, d_matl );
    d_RMCRT->setBC<double, double>( abskg,  d_compAbskgLabel->getName(), patch, d_matl );
  }
  delete archive;
}
//______________________________________________________________________
//
//______________________________________________________________________
void RMCRT_Test::sched_initProperties( const LevelP& finestLevel,
                                        SchedulerP& sched,
                                        const int radCalc_freq )
{
  // Move the labels forward.    They were computed in initialize()
  //  This mimics what the component will handoff to RMCRT.
  d_RMCRT->sched_CarryForward_Var( finestLevel, sched, d_cellTypeLabel );
  d_RMCRT->sched_CarryForward_Var( finestLevel, sched, d_compAbskgLabel );
  d_RMCRT->sched_CarryForward_Var( finestLevel, sched, d_colorLabel );
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
