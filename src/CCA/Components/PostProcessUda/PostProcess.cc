/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/PostProcessUda/PostProcess.h>
#include <CCA/Components/PostProcessUda/ModuleFactory.h>

#include <Core/Grid/SimpleMaterial.h>
#include <Core/Util/DOUT.hpp>
#include <iomanip>

using namespace std;
using namespace Uintah;
Dout dbg_pp("postProcess", false);

//______________________________________________________________________
//
PostProcessUda::PostProcessUda( const ProcessorGroup * myworld,
			           const SimulationStateP sharedState,
                                const string         & udaDir ) :
  ApplicationCommon( myworld, sharedState ),
  d_udaDir(udaDir),
  d_dataArchive(0),
  d_timeIndex(0)
{}

//______________________________________________________________________
//
PostProcessUda::~PostProcessUda()
{
  delete d_dataArchive;

  for (unsigned int i = 0; i < d_savedLabels.size(); i++){
    VarLabel::destroy(d_savedLabels[i]);
  }

  if(d_Modules.size() != 0){
    vector<Module*>::iterator iter;
    for( iter  = d_Modules.begin();iter != d_Modules.end(); iter++){
      delete *iter;
    }
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
  proc0cout << "      timestep in the original uda is processed. \n";
  proc0cout << "    - Use a different uda name for the modifed uda to prevent confusion with the original uda.\n\n";
  proc0cout << "    - In the timestep.xml files the follow non-essential entries will be changed:\n";
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
  d_oneMatl = scinew SimpleMaterial();
  m_sharedState->registerSimpleMaterial( d_oneMatl );

  delt_label = getDelTLabel();

  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!d_dataArchiver){
    throw InternalError("postProcessUda:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  //  Find timestep data from the original uda
  d_dataArchive = scinew DataArchive( d_udaDir, d_myworld->myRank(), d_myworld->nRanks() );
  d_dataArchive->queryTimesteps( d_timesteps, d_times );
  d_dataArchive->turnOffXMLCaching();

  // try to add a time to d_times that will get us passed the end of the
  // simulation -- yes this is a hack!!
  d_times.push_back(10 * d_times[d_times.size() - 1]);
  d_timesteps.push_back(999999999);

  proc0cout << "Time information from the original uda\n";
  for (unsigned int t = 0; t< d_timesteps.size(); t++ ){
    proc0cout << " *** timesteps " << d_timesteps[t] << " times: " << d_times[t] << endl;
  }

  //__________________________________
  //  define the varLabels that will be saved
  vector<string> varNames;
  vector< const TypeDescription *> typeDescriptions;
  d_dataArchive->queryVariables( varNames, typeDescriptions );

  proc0cout << "\nLabels discovered in the original uda\n";
  for (unsigned int i = 0; i < varNames.size(); i++) {
    d_savedLabels.push_back( VarLabel::create( varNames[i], typeDescriptions[i] ) );
    proc0cout << " *** Label: " << varNames[i] << endl;
  }

  //__________________________________
  //  create the analysis modules
  d_Modules = ModuleFactory::create(prob_spec, m_sharedState, m_output, d_dataArchive);

  vector<Module*>::iterator iter;
  for( iter  = d_Modules.begin(); iter != d_Modules.end(); iter++) {
    Module* m = *iter;
    m->problemSetup();
  }

  proc0cout << "\n";

}

//______________________________________________________________________
//
void PostProcessUda::scheduleInitialize(const LevelP& level,
                                        SchedulerP& sched)
{
  Task* t = scinew Task("PostProcessUda::initialize", this,
                       &PostProcessUda::initialize);

  t->computes( delt_label, level.get_rep() );

  GridP grid = level->getGrid();
  d_lb = sched->getLoadBalancer();

  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  t->setType(Task::OncePerProc);

  sched->addTask( t, perProcPatches, m_sharedState->allMaterials() );

  vector<Module*>::iterator iter;
  for( iter  = d_Modules.begin(); iter != d_Modules.end(); iter++){
    Module* m = *iter;
    m->scheduleInitialize( sched, level);
  }


}

//______________________________________________________________________
//
void PostProcessUda::scheduleRestartInitialize(const LevelP& level,
                                               SchedulerP& sched)
{
}

//______________________________________________________________________
//
void PostProcessUda::initialize(const ProcessorGroup* pg,
			           const PatchSubset* patches,
			           const MaterialSubset* /*matls*/,
			           DataWarehouse* /*old_dw*/,
			           DataWarehouse* new_dw)
{
  delt_vartype delt_var = 0.0;
  new_dw->put( delt_var, delt_label );
}

//______________________________________________________________________
//  This task is only called once.
void  PostProcessUda::scheduleTimeAdvance( const LevelP& level,
                                           SchedulerP& sched )
{
  sched_readDataArchive( level, sched );

  vector<Module*>::iterator iter;
  for( iter  = d_Modules.begin(); iter != d_Modules.end(); iter++){
    Module* m = *iter;
    m->scheduleDoAnalysis( sched, level);
  }
}

//______________________________________________________________________
//    Schedule for each patch that this processor owns.
//    The DataArchiver::output() will cherry pick the variables to output

void PostProcessUda::sched_readDataArchive(const LevelP& level,
                                           SchedulerP& sched)
{
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  const PatchSubset* patches = perProcPatches->getSubset(d_myworld->myRank());


  Task* t = scinew Task("PostProcessUda::readDataArchive", this,
                        &PostProcessUda::readDataArchive);


  // manually determine which matls to use in scheduling.
  // The sharedState does not have any materials registered
  // so you have to do it manually
  MaterialSetP allMatls = scinew MaterialSet();
  allMatls->createEmptySubsets(1);
  MaterialSubset* allMatlSubset = allMatls->getSubset(0);

  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;

  //__________________________________
  //  Loop over all saved labels
  for (unsigned int i = 0; i < d_savedLabels.size(); i++) {
    VarLabel* label = d_savedLabels[i];
    string labelName = label->getName();

    // find the range of matls over these patches
    ConsecutiveRangeSet matlsRangeSet;

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      ConsecutiveRangeSet matls = d_dataArchive->queryMaterials(labelName, patch, d_timeIndex);
      matlsRangeSet = matlsRangeSet.unioned(matls);
    }

    //__________________________________
    // Compute the material set and subset
    MaterialSet* matlSet;
    if (prevMatlSet != nullptr && prevRangeSet == matlsRangeSet) {
      matlSet = prevMatlSet.get_rep();
    }
    else {

      matlSet = scinew MaterialSet();
      vector<int> matls_vec;
      matls_vec.reserve(matlsRangeSet.size());

      for (ConsecutiveRangeSet::iterator iter = matlsRangeSet.begin(); iter != matlsRangeSet.end(); iter++) {
        if ( !allMatlSubset->contains(*iter) ) {
          allMatlSubset->add(*iter);
        }
	 matls_vec.push_back(*iter);
      }

      matlSet->addAll(matls_vec);
      prevRangeSet = matlsRangeSet;
      prevMatlSet  = matlSet;
    }

    allMatlSubset->sort();

    //__________________________________
    // schedule the computes for each patch that
    // this processor owns. The DataArchiver::output task
    // will then pick and choose which variables to write based on the input file
    t->computes(label, patches, matlSet->getUnion());
  }  // loop savedLabels

  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls.get_rep());

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
  double time = d_times[d_timeIndex];
  int timestep = d_timesteps[d_timeIndex];
  proc0cout << "*** working on timestep: " << timestep << " physical time: " << time << endl;

#if 0
  old_dw->unfinalize();
  d_dataArchive->postProcess_ReadUda(pg, d_timeIndex-1, d_oldGrid, patches, old_dw, d_lb);
  old_dw->refinalize();
#endif

  d_dataArchive->postProcess_ReadUda(pg, d_timeIndex, d_oldGrid, patches, new_dw, d_lb);
  d_timeIndex++;
}


//______________________________________________________________________
//
void PostProcessUda::scheduleComputeStableTimeStep(const LevelP& level,
                                                   SchedulerP& sched)
{
  Task* t = scinew Task("PostProcessUda::computeDelT",
                  this, &PostProcessUda::computeDelT);

  t->computes( delt_label, level.get_rep() );

  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);

  t->setType(Task::OncePerProc);
  sched->addTask( t, perProcPatches, m_sharedState->allMaterials() );

}

//______________________________________________________________________
//    This timestep is used to increment the physical time
void PostProcessUda::computeDelT(const ProcessorGroup*,
                                 const PatchSubset*,
                                 const MaterialSubset*,
                                 DataWarehouse* /*old_dw*/,
                                 DataWarehouse* new_dw)

{
  ASSERT( d_timeIndex >= 0);

  double delt = d_times[d_timeIndex];

  if ( d_timeIndex > 0 ){
    delt = d_times[d_timeIndex] - d_times[d_timeIndex-1];
  }

  delt_vartype delt_var = delt;

  new_dw->put(delt_var, delt_label);
  proc0cout << "*** delT (" << delt << ")" <<endl;
}

//______________________________________________________________________
//    Returns the physical time of the last output
double PostProcessUda::getMaxTime()
{
  if (d_times.size() <= 1){
    return 0;
  }else {
    return d_times[d_times.size()-2]; // the last one is the hacked one, see problemSetup
  }
}
//______________________________________________________________________
//    Returns the physical time of the first output
double PostProcessUda::getInitialTime()
{
  if (d_times.size() <= 1){
    return 0;
  }else {
    return d_times[0];
  }
}

//______________________________________________________________________
//  If the number of materials on a level changes or if the grid
//  has changed then call for a recompile
bool
PostProcessUda::needRecompile( const double   /* time */,
                               const double   /* dt */,
                               const GridP  & /* grid */ )
{
  bool recompile = d_gridChanged;
  d_gridChanged = false;   // reset flag

  int numLevels = d_oldGrid->numLevels();
  vector<int> level_numMatls( numLevels );

  for (int L = 0; L < numLevels; L++) {
    level_numMatls[L] = d_dataArchive->queryNumMaterials(*d_oldGrid->getLevel(L)->patchesBegin(), d_timeIndex);

    if (L >=(int) d_numMatls.size() || d_numMatls[L] != level_numMatls[L] ) {
      recompile = true;
    }
  }

  d_numMatls = level_numMatls;

/*`==========TESTING==========*/
  recompile = true;      // recompile the taskgraph every timestep
                         // If the number of saved variables changes or the number of matls on a level then
                         // you need to recompile.
/*===========TESTING==========`*/
  return recompile;
}

//______________________________________________________________________
// called by the SimulationController once per timestep
GridP PostProcessUda::getGrid()
{
  GridP newGrid = d_dataArchive->queryGrid(d_timeIndex);

  if (d_oldGrid == nullptr || !(*newGrid.get_rep() == *d_oldGrid.get_rep())) {
    d_gridChanged = true;
    d_oldGrid = newGrid;
    d_lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return d_oldGrid;
}




