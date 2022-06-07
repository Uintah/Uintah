/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/PostProcessUda/PostProcessUda.h>
#include <CCA/Components/PostProcessUda/ModuleFactory.h>
#include <Core/Exceptions/InternalError.h>

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>

#include <Core/Grid/SimpleMaterial.h>
#include <Core/Util/DOUT.hpp>
#include <iomanip>

using namespace std;
using namespace Uintah;
Dout dbg_pp("postProcess", "PostProcessUda", "PostProcessUda debug info", false);

//______________________________________________________________________
//
PostProcessUda::PostProcessUda( const ProcessorGroup * myworld,
                                const MaterialManagerP materialManager,
                                const string         & udaDir ) :
  ApplicationCommon( myworld, materialManager ),
  d_udaDir(udaDir)
{}

//______________________________________________________________________
//
PostProcessUda::~PostProcessUda()
{
  delete d_dataArchive;

  for (unsigned int i = 0; i < d_udaSavedLabels.size(); i++){
    VarLabel::destroy(d_udaSavedLabels[i]);
  }

  if(d_Modules.size() != 0){
    for( auto iter  = d_Modules.begin();iter != d_Modules.end(); iter++){
      delete *iter;
    }
  }
  
  for( auto iter  = d_analysisModules.begin(); iter != d_analysisModules.end(); iter++){
    AnalysisModule* am = *iter;
    am->releaseComponents();
    delete am;
  }
}
//______________________________________________________________________
//
void PostProcessUda::problemSetup(const ProblemSpecP& prob_spec,
                                  const ProblemSpecP& restart_ps,
                                  GridP& grid)
{
  //__________________________________
  //  Add a warning message
  proc0cout << "\n______________________________________________________________________\n";
  proc0cout << "                       P O S T P R O C E S S U D A \n\n";
  proc0cout << "    - If you're using this on a machine with a reduced set of system calls (mira) configure with\n";
  proc0cout << "          --with-boost\n";
  proc0cout << "      This will enable the non-system copy functions.\n\n";
  proc0cout << "    - You must manually copy all On-the-Fly files/directories from the original uda\n";
  proc0cout << "      to the new uda, postProcessUda ignores them.\n\n";
  proc0cout << "    - The <outputInterval>, <outputTimestepInterval> tags are ignored and every\n";
  proc0cout << "      timestep in the original uda is processed. \n\n";
  proc0cout << "    - Use a different uda name for the modifed uda to prevent confusion with the original uda.\n\n";
  proc0cout << "    - In the new.uda/timestep.xml the follow non-essential entries will be changed:\n";
  proc0cout << "           numProcs:      Number of procs used during the reduceUda run.\n";
  proc0cout << "           oldDelt:       Difference in timesteps, i.e., time(TS) - time (TS-1), in physical time.\n";
  proc0cout << "           proc:          The processor to patch assignment.\n\n";
  proc0cout << "    - The number of files inside of a timestep directory will now equal the number of processors used to reduce the uda\n";
  proc0cout << "      <<< You should use the same number of processors to reduce the uda as you will use to visualize it >>> \n\n";
  proc0cout << "      For large runs this should speed up data transfers and post processing utilities\n\n";
  proc0cout << "    - Checkpoint directories are copied with system calls from the original -> modified uda.\n";
  proc0cout << "      Only 1 processor is used during the copy so this could be slow for large checkpoints directories.\n";
  proc0cout << "      Consider moving this manually.\n\n";
  proc0cout << "______________________________________________________________________\n\n";

  //__________________________________
  //
  setLockstepAMR(true);
  m_output->setSwitchState(true);         /// HACK NEED TO CHANGE THIS

  // This matl is for delT
  SimpleMaterial * oneMatl = scinew SimpleMaterial();
  m_materialManager->registerSimpleMaterial( oneMatl );

  //__________________________________
  //  Find timestep data from the original uda
  d_dataArchive = scinew DataArchive( d_udaDir, d_myworld->myRank(), d_myworld->nRanks() );
  d_dataArchive->queryTimesteps( d_udaTimesteps, d_udaTimes );
  d_dataArchive->turnOffXMLCaching();

  proc0cout << "Time information from the original uda\n";
  for (unsigned int t = 0; t< d_udaTimesteps.size(); t++ ){
    proc0cout << " *** " << t << " sim timestep " << d_udaTimesteps[t] << " sim time: " << d_udaTimes[t] << endl;
  }


  //__________________________________
  //  define the varLabels that will be saved
  vector<string> varNames;
  vector<int>    numMatls;
  vector< const TypeDescription *> typeDescriptions;
  d_dataArchive->queryVariables( varNames, numMatls, typeDescriptions );

  proc0cout << "\nLabels discovered in the original uda\n";
  for (unsigned int i = 0; i < varNames.size(); i++) {
    d_udaSavedLabels.push_back( VarLabel::create( varNames[i], typeDescriptions[i] ) );
    proc0cout << " *** Label: " << varNames[i] << endl;
  }

  proc0cout << "\n";

  //__________________________________
  //  create the PostProcess analysis modules
  d_Modules = ModuleFactory::create(prob_spec, m_materialManager, m_output, d_dataArchive);

  for( auto iter  = d_Modules.begin(); iter != d_Modules.end(); iter++) {
    Module* m = *iter;
    m->problemSetup();
  }

  //__________________________________
  //  Set up data OnTheFly analysis modules
  d_analysisModules = AnalysisModuleFactory::create(d_myworld,
                                                    m_materialManager,
                                                    prob_spec);

  for( auto iter  = d_analysisModules.begin(); iter != d_analysisModules.end(); iter++) {
    AnalysisModule* am = *iter;
    std::vector<std::vector<const VarLabel* > > dummy;

    am->setComponents( dynamic_cast<ApplicationInterface*>( this ) );
    am->problemSetup(prob_spec, restart_ps, grid, dummy, dummy);
  }


  // Adjust the time state - done after it is read. If the values are
  // zero they will be ignored when checked.
  ApplicationCommon::setDelTOverrideRestart( 0 );
  ApplicationCommon::setDelTInitialMax(      0 );
  ApplicationCommon::setDelTInitialRange(    0 );
  
  ApplicationCommon::setDelTMin( 0 );
  ApplicationCommon::setDelTMax( 0 );
  ApplicationCommon::setDelTMultiplier( 1.0 );
  ApplicationCommon::setDelTMaxIncrease( 0.0 );
  
  ApplicationCommon::setSimTime(          d_udaTimes[0] );
  ApplicationCommon::setSimTimeMax(       d_udaTimes[d_udaTimes.size()-1] );
  ApplicationCommon::setSimTimeEndAtMax(  false );
  ApplicationCommon::setSimTimeClampToOutput( false );

  ApplicationCommon::setTimeStepsMax( d_udaTimes.size() );
}

//______________________________________________________________________
//
void PostProcessUda::scheduleInitialize(const LevelP& level,
                                        SchedulerP& sched)
{
  // Misc setup calls
  Dir fromDir( d_udaDir );
  m_output->postProcessUdaSetup( fromDir );

  //__________________________________
  //    PostProcess modules
  for( auto iter  = d_Modules.begin(); iter != d_Modules.end(); iter++){
    Module* m = *iter;
    m->scheduleInitialize( sched, level );
  }
  
  //__________________________________
  //    OnTheFly dataAnalysis
  for( auto iter  = d_analysisModules.begin(); iter != d_analysisModules.end(); iter++){
    AnalysisModule* am = *iter;
    am->scheduleInitialize( sched, level );
  }
}

//______________________________________________________________________
//  This task is only called once.
void  PostProcessUda::scheduleTimeAdvance( const LevelP& level,
                                           SchedulerP& sched )
{
  sched_readDataArchive( level, sched );

  //__________________________________
  //    PostProcess analysis
  for( auto iter  = d_Modules.begin(); iter != d_Modules.end(); iter++){
    Module* m = *iter;
    m->scheduleDoAnalysis( sched, level);
  }
  
  //__________________________________
  //    OnTheFly analysis
  for( auto iter  = d_analysisModules.begin(); iter != d_analysisModules.end(); iter++){
    AnalysisModule* am = *iter;
    am->scheduleDoAnalysis( sched, level);
  }
}

//______________________________________________________________________
//    Schedule for each patch that this processor owns.
//    The DataArchiver::output() will cherry pick the variables to output
void PostProcessUda::sched_readDataArchive(const LevelP& level,
                                           SchedulerP& sched)
{
  Task* t = scinew Task("PostProcessUda::readDataArchive", this,
                        &PostProcessUda::readDataArchive);

  t->requires(Task::OldDW, getTimeStepLabel());
  
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = m_loadBalancer->getPerProcessorPatchSet(grid);
  const PatchSubset* patches = perProcPatches->getSubset(d_myworld->myRank());

  vector<int> allMatls_vec;
  //__________________________________
  //  Loop over all saved labels
  for (unsigned int i = 0; i < d_udaSavedLabels.size(); i++) {
    VarLabel* label = d_udaSavedLabels[i];
    string labelName = label->getName();

    // find the range of matls over these patches
    ConsecutiveRangeSet matlsRangeSet;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);
      ConsecutiveRangeSet matls = d_dataArchive->queryMaterials(labelName, patch, d_simTimestep);
      matlsRangeSet = matlsRangeSet.unioned(matls);
    }

    //__________________________________
    // Compute the material set
    vector<int> matls_vec;
    matls_vec.reserve(matlsRangeSet.size());

    for (ConsecutiveRangeSet::iterator iter = matlsRangeSet.begin(); iter != matlsRangeSet.end(); iter++) {
      matls_vec.push_back(*iter);
      allMatls_vec.push_back(*iter);
    }

    MaterialSet * matlSet = scinew MaterialSet();
    matlSet->addAll(matls_vec);

    //__________________________________
    // schedule the computes for each patch that
    // this processor owns. The DataArchiver::output task
    // will then pick and choose which variables to write to the new uda based on the input file
    t->computes(label, patches, matlSet->getUnion());
    delete matlSet;
   
  }  // loop savedLabels

  // Create material set without duplicate entries
  MaterialSet * allMatls = scinew MaterialSet();
  allMatls->addAll_unique(allMatls_vec);
  allMatls->addReference();

  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls );
  
  if (allMatls && allMatls->removeReference()){
    delete allMatls;
  }

}

//______________________________________________________________________
//  This task reads data from the dataArchive and 'puts' it into the data Warehouse
//
void PostProcessUda::readDataArchive(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  old_dw->get( timeStep, getTimeStepLabel() );
  
  double time  = d_udaTimes[d_simTimestep];
  int udaTimestep = d_udaTimesteps[d_simTimestep];
  proc0cout << "    *** working on uda timestep: " << udaTimestep << " simTimestep: " <<  timeStep - 1 << " physical time: " << time << endl;

  // populate the old_dw with variables from the uda.
  // Only one timestep

  int old_dw_timestep = NOTUSED;
  bool isSet    = false;

  for( auto iter  = d_Modules.begin(); iter != d_Modules.end(); iter++){
    Module* m = *iter;
    int tmp = m->getTimestep_OldDW();

    // bulletproofing
    if ( isSet && tmp != old_dw_timestep ){
      ostringstream err;
      err << "ERROR: PostProcessUda::readDataArchive.  The module (" << m->getName() << ") "
          << "requested data from a previous timestep.  You can only specify one timestep for all modules.\n";
      throw InternalError( err.str(), __FILE__, __LINE__ );
    }
    
    if ( isSet == false && tmp != NOTUSED ){
      old_dw_timestep = tmp;
      isSet = true;
    }
  }
  
  proc0cout << "    Reading data archive " << endl;


  const Level * level = getLevel(patches);
  const GridP grid    = level->getGrid();
    
  if( old_dw_timestep != NOTUSED && udaTimestep >= old_dw_timestep && udaTimestep > 1){
    
   proc0cout << "    OLD_DW  ";
   old_dw->unfinalize();
   d_dataArchive->postProcess_ReadUda(pg, old_dw_timestep, grid, patches, old_dw, m_loadBalancer);
   old_dw->refinalize();
  }

  // new dw
  proc0cout << "    NEW_DW\n";
  d_dataArchive->postProcess_ReadUda(pg, d_simTimestep, grid, patches, new_dw, m_loadBalancer);
  d_simTimestep++;

  proc0cout << "    __________________________________ " << endl;
  
  // new_dw->print();
  // old_dw->print();
}


//______________________________________________________________________
//
void PostProcessUda::scheduleComputeStableTimeStep(const LevelP& level,
                                                   SchedulerP& sched)
{
  Task* t = scinew Task("PostProcessUda::computeDelT",
                  this, &PostProcessUda::computeDelT);

  t->computes( getDelTLabel(), level.get_rep() );

  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = m_loadBalancer->getPerProcessorPatchSet(grid);

  t->setType(Task::OncePerProc);
  sched->addTask( t, perProcPatches, m_materialManager->allMaterials() );
}

//______________________________________________________________________
//    This timestep is used to increment the physical time
void PostProcessUda::computeDelT(const ProcessorGroup*,
                                 const PatchSubset*,
                                 const MaterialSubset*,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)

{
  ASSERT( d_simTimestep >= 0 );

  double delt;

  // For time step 0 the sim time will be 0 so the delT will simply be
  // the value of the first sim time. After that it will be the
  // differential between the time steps. Until the last which is moot.
  if ( d_simTimestep == 0 ) {
    delt = d_udaTimes[d_simTimestep];    
  } 
  else if ( d_simTimestep < (int) d_udaTimes.size() ) {
    delt = d_udaTimes[d_simTimestep] - d_udaTimes[d_simTimestep-1];
  } 
  else {
    delt = 1e99;
  }

  new_dw->put(delt_vartype(delt), getDelTLabel());
}

//______________________________________________________________________
//    Returns the physical time of the last output
double PostProcessUda::getMaxTime()
{
  if (d_udaTimes.size() <= 1){
    return 0;
  }else {
    return d_udaTimes[d_udaTimes.size()-2]; // the last one is the hacked one, see problemSetup
  }
}

//______________________________________________________________________
//    Returns the physical time of the first output
double PostProcessUda::getInitialTime()
{
  if (d_udaTimes.size() <= 1){
    return 0;
  }
  else {
    return d_udaTimes[0];
  }
}

//______________________________________________________________________
//  If the number of materials on a level changes or if the grid
//  has changed then call for a recompile
bool
PostProcessUda::needRecompile( const GridP& currentGrid )
{

#if 0  // is this needed --Todd
  int numLevels = currentGrid->numLevels();
  vector<int> level_numMatls( numLevels );
  d_numMatls = level_numMatls;
  
  for (int L = 0; L < numLevels; L++) {
    level_numMatls[L] = d_dataArchive->queryNumMaterials(*d_oldGrid->getLevel(L)->patchesBegin(), d_simTimestep);

    if (L >=(int) d_numMatls.size() || d_numMatls[L] != level_numMatls[L] ) {
      recompile = true;
    }
  }
#endif  

  bool recompile = true;  // recompile the taskgraph every timestep
                         // If the number of saved variables changes or the number of matls on a level then
                         // you need to recompile.
  return recompile;
}

//______________________________________________________________________
// called by the SimulationController once per timestep
GridP PostProcessUda::getGrid( const GridP& currentGrid)
{
  GridP newGrid = d_dataArchive->queryGrid(d_simTimestep);

  if (currentGrid == nullptr || !(*newGrid.get_rep() == *currentGrid.get_rep())) {
  
    m_loadBalancer->possiblyDynamicallyReallocate(newGrid, true);
    proc0cout << "    Grid has changed \n";
    return newGrid;
  } 
  else{
    proc0cout << "    Grid has not changed \n";
    return currentGrid;
  }
  return nullptr;
}
