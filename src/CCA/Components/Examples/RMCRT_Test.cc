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
#include <CCA/Components/Regridder/PerPatchVars.h>
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
#include <Core/Grid/AMR.h>
#include <Core/Grid/AMR_CoarsenRefine.h>
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
#include <Core/Grid/DbgOutput.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

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
  //d_examplesLabel = scinew ExamplesLabel();
  d_colorLabel   = VarLabel::create("color",    CCVariable<double>::getTypeDescription());           
  d_divQLabel    = VarLabel::create("divQ",     CCVariable<double>::getTypeDescription());           
  d_abskgLabel   = VarLabel::create("abskg",    CCVariable<double>::getTypeDescription());           
  d_absorpLabel  = VarLabel::create("absorp",   CCVariable<double>::getTypeDescription());
  d_sigmaT4Label = VarLabel::create("sigmaT4",  CCVariable<double>::getTypeDescription()); 
   
  d_gac = Ghost::AroundCells;
  d_gn  = Ghost::None;
  d_matl = 0;
  d_doFakeRMCRT = false;
  d_doRealRMCRT = false;
  d_initColor = -9;
  d_initAbskg = -9;
  d_orderOfInterpolation = -9;
  d_CoarseLevelRMCRTMethod = true;
  d_multiLevelRMCRTMethod  = false;
}
//______________________________________________________________________
//
RMCRT_Test::~RMCRT_Test ( void )
{
  if ( d_realRMCRT ) 
    delete d_realRMCRT;
    
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

  BBox gridBoundingBox;
  grid->getSpatialRange( gridBoundingBox );
  d_gridMax = gridBoundingBox.max().asVector();
  d_gridMin = gridBoundingBox.min().asVector();
  d_centerOfDomain   = (( d_gridMax - d_gridMin ) / 2.0 ) + d_gridMin;

  //defaults
  d_radiusOfBall     = 0.10 * d_gridMax.x();
  d_radiusOfOrbit    = 0.25 * d_gridMax.x();
  d_angularVelocity  = 10;
  

  //__________________________________
  //  pseudo RMCRT
  if (prob_spec->findBlock("pseudoRMCRT")){
    d_doFakeRMCRT  = true;
    ProblemSpecP spec = prob_spec->findBlock("pseudoRMCRT");
    spec->get("ballRadius",             d_radiusOfBall);
    spec->get("orbitRadius",            d_radiusOfOrbit);
    spec->get("angularVelocity",        d_angularVelocity);
    cout << "__________________________________ Reading in pseudoRMCRT section of ups file" << endl;
  } 
  
  //__________________________________
  //  Real RMCRT
  if (prob_spec->findBlock("RMCRT")){
    d_doRealRMCRT  = true;
    ProblemSpecP rmcrt_db = prob_spec->findBlock("RMCRT"); 
    
    d_realRMCRT = scinew Ray(); 
    d_realRMCRT->registerVarLabels(0,
                                 d_abskgLabel,
                                 d_absorpLabel,
                                 d_colorLabel,
                                 d_divQLabel );
                                 
    rmcrt_db->require("Temperature",  d_initColor);
    rmcrt_db->require("abskg",        d_initAbskg);
    
    d_realRMCRT->problemSetup( rmcrt_db );
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

    rmcrt_ps->require( "orderOfInterpolation", d_orderOfInterpolation);

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
  //  If the RMCRT is performed on each fine patch with a halo of coarse patches
  //  around the fine patch
  if(d_multiLevelRMCRTMethod){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();

    // compute Radiative properties on the finest level
    if(d_doRealRMCRT){
      int time_sub_step = 0;
      d_realRMCRT->sched_initProperties( fineLevel, sched, time_sub_step );
    }

    // coarsen data to the coarser levels
    for (int l = 0; l < maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      scheduleCoarsenAll (level, sched);
      d_realRMCRT->sched_setBoundaryConditions( level, sched );
    }
    
    // only schedule RMCRT and pseudoCFD on the finest level    
    scheduleShootRays_multiLevel( sched, fineLevel,     matls );
    schedulePseudoCFD(            sched, finestPatches, matls );
  }
  
  //______________________________________________________________________
  //   2 - L E V E L   A P P R O A C H
  //  If the RMCRT is performed on only the coarse level
  // and the results are interpolated to the fine level
  if(d_CoarseLevelRMCRTMethod){
    const LevelP& fineLevel = grid->getLevel(maxLevels-1);
    const PatchSet* finestPatches = fineLevel->eachPatch();
   
    // compute Radiative properties on the finest level
    if(d_doRealRMCRT){
      int time_sub_step = 0;
      d_realRMCRT->sched_initProperties( fineLevel, sched, time_sub_step );
    }
    
    
    for (int l = 0; l <= maxLevels-1; l++) {
      const LevelP& level = grid->getLevel(l);
      scheduleCoarsenAll (level, sched);
      scheduleShootRays_onCoarseLevel( sched, level, matls );
    }

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      scheduleRefine_Q (sched,  patches, matls);
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
      d_realRMCRT->setBC(color,  d_colorLabel->getName(), patch, d_matl);
    }
  }
}


//______________________________________________________________________
//
void RMCRT_Test::scheduleShootRays_multiLevel(SchedulerP& sched,
                                              const LevelP& level,
                                              const MaterialSet* matls)
{
  printSchedule(level,dbg,"RMCRT_Test::scheduleShootRays_multiLevel");
  
  Task* t = scinew Task("RMCRT_Test::shootRays_multiLevel",
                  this, &RMCRT_Test::shootRays_multiLevel);

  if (d_doFakeRMCRT){
    Task::DomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    t->requires(Task::OldDW, d_colorLabel,  d_gn, 0);
    t->requires(Task::OldDW, d_colorLabel,  allPatches, Task::CoarseLevel,allMatls, ND, d_gn, 0);

    t->computes( d_divQLabel );
    sched->addTask(t, level->eachPatch(), matls);
  }
  
  if(d_doRealRMCRT){
    Task::WhichDW abskg_dw   = Task::NewDW;
    Task::WhichDW sigmaT4_dw = Task::NewDW;
    bool modifies_divQ       = false;
    d_realRMCRT->sched_rayTrace_dataOnion(level,sched, abskg_dw, sigmaT4_dw,modifies_divQ);
  }
  
}
//______________________________________________________________________
void RMCRT_Test::shootRays_multiLevel ( const ProcessorGroup*,
                                        const PatchSubset* finePatches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw )
{ 
  const Level* fineLevel = getLevel(finePatches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  
  
  IntVector C_lo,C_hi;
  coarseLevel->findInteriorCellIndexRange(C_lo, C_hi);
  
  cout << " getting coarse level data: C_lo " << C_lo << " C_hi: " << C_hi << endl;
  constCCVariable<double> colorCoarse;
  old_dw->getRegion(colorCoarse, d_colorLabel,  d_matl, coarseLevel, C_lo, C_hi);

  for(int p=0;p<finePatches->size();p++){
    const Patch* finePatch = finePatches->get(p);

    printTask(finePatches, finePatch,dbg,"shootRays_multiLevel");

    CCVariable<double> sumColorDiff;
    constCCVariable<double> colorFine;
    new_dw->allocateAndPut(sumColorDiff, d_divQLabel, d_matl, finePatch);
    old_dw->get(colorFine,               d_colorLabel,d_matl, finePatch, d_gn, 0);
    
    sumColorDiff.initialize(0.0);
    
    IntVector P_lo = finePatch->getCellLowIndex();
    IntVector P_hi = finePatch->getCellHighIndex();
    

    for (CellIterator iter = finePatch->getCellIterator();!iter.done();iter++){
      const IntVector& f_cell = *iter;
      int i = -9;
      int j = -9;
      int k = -9;
      
      //__________________________________
      // Ray in x dir       fine patch                        
      j = f_cell.y();
      k = f_cell.z();
      for( i=P_lo.x(); i<P_hi.x(); i++ ){      
        IntVector h (i,j,k);                     
        //cout << "    fineLevel X: " << h << endl;             
        sumColorDiff[f_cell] += colorFine[h] - colorFine[f_cell];  
      }
      // Ray in y dir       fine patch                        
      i = f_cell.x();
      k = f_cell.z();
      for( j=P_lo.y(); j<P_hi.y(); j++ ){      
        IntVector h (i,j,k);                     
        //cout << "    fineLevel Y: " << h << endl;             
        sumColorDiff[f_cell] += colorFine[h] - colorFine[f_cell];  
      }
#if 0
      // Ray in z dir       fine patch                        
      i = f_cell.x();
      j = f_cell.y();
      for( k=P_lo.z(); k<P_hi.z(); k++ ){      
        IntVector h (i,j,k);                     
        //cout << "    fineLevel Z: " << h << endl;             
        sumColorDiff[f_cell] += colorFine[h] - colorFine[f_cell];  
      }
#endif      
      
      //__________________________________
      // Ray in x dir        coarse level   Don't add contribution if under the fine patch.
      IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
      j = c_cell.y();                             
      k = c_cell.z(); 
      
      for( i=C_lo.x(); i<C_hi.x(); i++ ){      
        IntVector h (i,j,k);  
        IntVector h_fine = coarseLevel->mapCellToFiner(h);
        bool testHi = (h_fine.x() > P_hi.x() - 1 );  // P_hi is exclusive, we want inclusive
        bool testLo = (h_fine.x() < P_lo.x() ); 
        
        if( testHi|| testLo ){         // possibly replace conditional with one_or_zero    
          //cout << "    coarseLevel X: " << h << endl;
          sumColorDiff[f_cell] += colorCoarse[h] - colorFine[f_cell];  
        }
      }
                                                   
      // Ray in y dir        coarse level
      i = c_cell.x();         
      k = c_cell.z(); 
      
      for( j=C_lo.y(); j<C_hi.y(); j++ ){      
        IntVector h (i,j,k);  
        IntVector h_fine = coarseLevel->mapCellToFiner(h);
        bool testHi = (h_fine.y() > P_hi.y() - 1 ); 
        bool testLo = (h_fine.y() < P_lo.y() );
        
        if( testHi|| testLo ){         
          //cout << "    coarseLevel Y: " << h << endl;
          sumColorDiff[f_cell] += colorCoarse[h] - colorFine[f_cell];  
        }
      }                                          

#if 0
      // Ray in z dir        coarse level
      i = c_cell.x();         
      j = c_cell.y(); 
      
      for( k=C_lo.z(); k<C_hi.z(); k++ ){      
        IntVector h (i,j,k);  
        IntVector h_fine = coarseLevel->mapCellToFiner(h);
        bool testHi = (h_fine.z() > P_hi.z() - 1 ); 
        bool testLo = (h_fine.z() < P_lo.z() );
        
        if( testHi|| testLo ){         // possibly replace conditional with one_or_zero    
          cout << "    coarseLevel Z: " << h << endl;
          sumColorDiff[f_cell] += colorCoarse[h] - colorFine[f_cell];  
        }
      }
#endif
    }  // fine level cell iterator
  }  // fine patch loop
}  

//______________________________________________________________________
//
void RMCRT_Test::scheduleShootRays_onCoarseLevel(SchedulerP& sched,
                                                 const LevelP& level,
                                                 const MaterialSet* matls)
{
  GridP grid = level->getGrid();
  int maxLevels = level->getGrid()->numLevels();
  
  if(level->hasFinerLevel() || maxLevels == 1){
    printSchedule(level,dbg,"RMCRT_Test::scheduleShootRays_onCoarseLevel");

    //__________________________________
    //  
    if (d_doFakeRMCRT){
      Ghost::GhostType  gn  = Ghost::None;
      Task::DomainSpec  ND  = Task::NormalDomain;
      #define allPatches 0
      #define allMatls 0
      
      Task* t = scinew Task("RMCRT_Test::shootRays_onCoarseLevel",
                      this, &RMCRT_Test::shootRays_onCoarseLevel);
      
      t->requires(Task::OldDW, d_colorLabel,  allPatches, ND,allMatls, ND, gn,0);
      t->computes( d_divQLabel );
      sched->addTask(t, level->eachPatch(), matls);
    }

    if(d_doRealRMCRT){
      //d_realRMCRT->sched_sigmaT4(level,sched);
      
      Task::WhichDW abskg_dw   = Task::NewDW;
      Task::WhichDW sigmaT4_dw = Task::NewDW;
      bool modifies_divQ       = false;
      d_realRMCRT->sched_rayTrace(level,sched,abskg_dw,sigmaT4_dw,modifies_divQ);
    }  
  }
}
  
//______________________________________________________________________
void RMCRT_Test::shootRays_onCoarseLevel ( const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw )
{ 
  const Level* level = getLevel(patches);
  IntVector L_lo,L_hi;

  level->findInteriorCellIndexRange(L_lo, L_hi);
  constCCVariable<double> color;

  old_dw->getRegion(color, d_colorLabel,  d_matl, level, L_lo, L_hi);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"shootRays_onCoarseLevel");

    CCVariable<double> divQ;
    new_dw->allocateAndPut(divQ, d_divQLabel, d_matl, patch);
    divQ.initialize(0.0);
#if 0
    IntVector P_lo = patch->getCellLowIndex();
    IntVector P_hi = patch->getCellHighIndex();
    IntVector middle = P_lo + (P_hi - P_lo)/IntVector(2,2,2);
#endif

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      // Ray in x dir                            
      int j = c.y();                             
      int k = c.z();                             
      for(int i=L_lo.x(); i<L_hi.x(); i++){      
        IntVector h (i,j,k);                     
        //cout << "    " << h << endl;             
        divQ[c] += color[h] - color[c];  
      }                                          
      // Ray in Y dir                            
      int i = c.x();                             
      k = c.z();                                 
      for(int j=L_lo.y(); j<L_hi.y(); j++){      
        IntVector h (i,j,k);                     
        divQ[c] += color[h] - color[c];  
      }                                          

      // Ray in Z dir                            
      i = c.x();                                 
      j = c.y();                                 
      for(int k=L_lo.z(); k<L_hi.z(); k++){      
        IntVector h (i,j,k);                     
        divQ[c] += color[h] - color[c];  
      } 
    }
  }
}  
//______________________________________________________________________
//
void RMCRT_Test::scheduleRefine_Q(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls)
{
  const Level* fineLevel = getLevel(patches);
  int L_indx = fineLevel->getIndex();
  
  if(L_indx > 0 ){
     printSchedule(patches,dbg,"RMCRT_Test::scheduleRefine_Q (divQ)");

    Task* task = scinew Task("RMCRT_Test::refine_Q",this, 
                             &RMCRT_Test::refine_Q);
    
    Task::DomainSpec  ND  = Task::NormalDomain;
    #define allPatches 0
    #define allMatls 0
    task->requires(Task::NewDW, d_divQLabel, allPatches, Task::CoarseLevel, allMatls, ND, d_gn,0);
     
    task->computes(d_divQLabel);
    sched->addTask(task, patches, matls);
  }
}
  
//______________________________________________________________________
//
void RMCRT_Test::refine_Q(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse* new_dw)
{
  const Level* fineLevel = getLevel(patches);
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
  
  for(int p=0;p<patches->size();p++){  
    const Patch* finePatch = patches->get(p);
    printTask(patches, finePatch,dbg,"Doing refineQ");

    Level::selectType coarsePatches;
    finePatch->getCoarseLevelPatches(coarsePatches);

    CCVariable<double> sumColorDiff_fine;
    new_dw->allocateAndPut(sumColorDiff_fine, d_divQLabel, d_matl, finePatch);
    sumColorDiff_fine.initialize(0);
    
    IntVector refineRatio = fineLevel->getRefinementRatio();

    // region of fine space that will correspond to the coarse we need to get
    IntVector cl, ch, fl, fh;
    IntVector bl(0,0,0);  // boundary layer or padding
    int nghostCells = 1;
    bool returnExclusiveRange=true;
    
    getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl, 
                        nghostCells, returnExclusiveRange);

    dbg <<" refineQ: " 
        <<" finePatch  "<< finePatch->getID() << " fl " << fl << " fh " << fh
        <<" coarseRegion " << cl << " " << ch <<endl;

    constCCVariable<double> sumColorDiff_coarse;
    new_dw->getRegion(sumColorDiff_coarse, d_divQLabel, d_matl, coarseLevel, cl, ch);

    selectInterpolator(sumColorDiff_coarse, d_orderOfInterpolation, coarseLevel, fineLevel,
                       refineRatio, fl, fh,sumColorDiff_fine);

  }  // fine patch loop 
}
  
//______________________________________________________________________
void RMCRT_Test::scheduleErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
{
  printSchedule(level,dbg,"RMCRT_Test::errorEstimate");

  Task* task = scinew Task( "RMCRT_Test::errorEstimate", this, 
                            &RMCRT_Test::errorEstimate, false );

  task->requires( Task::NewDW, d_colorLabel,    d_gn, 0 );

  task->modifies( d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials() );
  task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );

  scheduler->addTask( task, scheduler->getLoadBalancer()->getPerProcessorPatchSet(level), d_sharedState->allMaterials() );
}

//______________________________________________________________________
void RMCRT_Test::scheduleInitialErrorEstimate ( const LevelP& level, SchedulerP& scheduler )
{
  printSchedule(level,dbg,"RMCRT_Test::scheduleInitialErrorEstimate");

  Task* task = scinew Task( "RMCRT_Test::initialErrorEstimate", this, 
                            &RMCRT_Test::errorEstimate, true );

  task->requires( Task::NewDW, d_colorLabel, d_gn, 0 );
  task->modifies( d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials() );
  task->modifies( d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials() );

  scheduler->addTask( task, level->eachPatch(), d_sharedState->allMaterials() );
}
//______________________________________________________________________
void RMCRT_Test::scheduleCoarsen ( const LevelP& coarseLevel, SchedulerP& scheduler )
{
}
//______________________________________________________________________
void RMCRT_Test::scheduleRefine ( const PatchSet* patches, 
                                  SchedulerP& scheduler )
{
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

  d_centerOfBall     = d_centerOfDomain;

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
        
        if( d_doFakeRMCRT ){
          Vector whereThisCellIs( patch->cellPosition( idx ) );
          Vector distanceToCenterOfDomain = whereThisCellIs - d_centerOfBall;
          color[idx] = distanceToCenterOfDomain.length();
        }
        
        if( d_doRealRMCRT ){
          color[idx] = d_initColor;
          abskg[idx] = d_initAbskg;
        }
      }
       // set boundary conditions 
      d_realRMCRT->setBC(color,  d_colorLabel->getName(), patch, matl);
      d_realRMCRT->setBC(abskg,  d_abskgLabel->getName(), patch, matl);
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
void RMCRT_Test::errorEstimate ( const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw, 
                                 bool initial )
{ 
  //__________________________________
  //   initial refinement region
  if(initial){

    const Level* level = getLevel(patches);

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      printTask(patches, patch,dbg,"Doing initialErrorEstimate");

      CCVariable<int> refineFlag;
      PerPatch<PatchFlagP> refinePatchFlag;
      new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch);
      new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(), 0, patch);

      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

      // loop over all the geometry objects
      for(int obj=0; obj<(int)d_refine_geom_objs.size(); obj++){
        GeometryPieceP piece = d_refine_geom_objs[obj]->getPiece();
        Vector dx = patch->dCell();

        int geom_level =  d_refine_geom_objs[obj]->getInitialData_int("level");

        //don't add refinement flags if the current level is greater than the geometry level specification
        if(geom_level!=-1 && level->getIndex()>=geom_level)
          continue;

        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          Point  lower  = patch->nodePosition(c);
          Vector upperV = lower.asVector() + dx; 
          Point  upper  = upperV.asPoint();

          if(piece->inside(upper) && piece->inside(lower))
            refineFlag[c] = true;
            refinePatch->set();
        }
      }  // object loop
    }  // patches loop
  }
  else{

    //__________________________________
    //   flag regions inside of the ball
    if ( getLevel(patches)->getIndex() == getLevel(patches)->getGrid()->numLevels()-1 ) {
      d_centerOfBall     = d_centerOfDomain;
    }

    for ( int p = 0; p < patches->size(); p++ ) {
      const Patch* patch = patches->get(p);

      printTask(patches, patch,dbg,"Doing errorEstimate");

      CCVariable<int> refineFlag;
      new_dw->getModifiable(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch);

      PerPatch<PatchFlagP> refinePatchFlag;
      new_dw->get(refinePatchFlag, d_sharedState->get_refinePatchFlag_label(), 0, patch);
      PatchFlag* refinePatch = refinePatchFlag.get().get_rep();

      bool foundErrorOnPatch = false;
      constCCVariable<double> color;
      new_dw->get( color,    d_colorLabel,    d_matl, patch, d_gn, 0 );

      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;

        if ( color[c] <= d_radiusOfBall ) {
          refineFlag[c]=true;
          foundErrorOnPatch = true;
        } else {
          refineFlag[c]=false;
        }
      }

      // flag this patch
      if ( foundErrorOnPatch ) {
        refinePatch->flag = true;
      } else {
        refinePatch->flag = false;
      }

    }  // patch loop
  }  // not initial timestep
}


//______________________________________________________________________
void RMCRT_Test::scheduleCoarsenAll( const LevelP& coarseLevel, 
                                     SchedulerP& sched )
{
  if(coarseLevel->hasFinerLevel()){
    printSchedule(coarseLevel,dbg,"RMCRT_Test::scheduleCoarsenAll");
    bool modifies = false;

    if(d_doFakeRMCRT){
      scheduleCoarsen_Q(coarseLevel, sched,Task::OldDW, modifies, d_colorLabel);
    }

    if(d_doRealRMCRT){
      modifies = false;
      scheduleCoarsen_Q(coarseLevel, sched, Task::NewDW, modifies, d_abskgLabel);
      scheduleCoarsen_Q(coarseLevel, sched, Task::NewDW, modifies, d_sigmaT4Label);
    }
  }
}

//______________________________________________________________________
void RMCRT_Test::scheduleCoarsen_Q ( const LevelP& coarseLevel, 
                                     SchedulerP& sched,
                                     Task::WhichDW this_dw,
                                     const bool modifies,
                                     const VarLabel* variable)
{ 
  string taskname = "        Coarsen_Q_" + variable->getName();
  printSchedule(coarseLevel,dbg,taskname);

  Task* t = scinew Task( taskname, this, &RMCRT_Test::coarsen_Q, 
                         variable, modifies, this_dw );
  
  if(modifies){
    t->modifies(variable);
  }else{
    t->requires(this_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0);
    t->computes(variable);
  }
  sched->addTask( t, coarseLevel->eachPatch(), d_sharedState->allMaterials() );
}

//______________________________________________________________________
void RMCRT_Test::coarsen_Q ( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw,
                             const VarLabel* variable,
                             const bool modifies,
                             Task::WhichDW which_dw )
{
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel = coarseLevel->getFinerLevel().get_rep();
  
  DataWarehouse* this_dw = new_dw;
  
  if( which_dw == Task::OldDW ){
    this_dw = old_dw;
  }
  

  for(int p=0;p<patches->size();p++){  
    const Patch* coarsePatch = patches->get(p);

    printTask(patches, coarsePatch,dbg,"Doing coarsen: " + variable->getName());

    // Find the overlapping regions...
    Level::selectType finePatches;
    coarsePatch->getFineLevelPatches(finePatches);

    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<double> Q_coarse;
      if(modifies){
        new_dw->getModifiable(Q_coarse,  variable, matl, coarsePatch);
      }else{
        new_dw->allocateAndPut(Q_coarse, variable, matl, coarsePatch);
      }
      Q_coarse.initialize(0.0);

      // coarsen
      bool computesAve = false;
      fineToCoarseOperator(Q_coarse,   computesAve, 
                           variable,   matl, new_dw,                   
                           coarsePatch, coarseLevel, fineLevel);        
    }
  }  // course patch loop 
}
} // namespace Uintah
