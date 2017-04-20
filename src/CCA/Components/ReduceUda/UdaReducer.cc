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

#include <CCA/Components/ReduceUda/UdaReducer.h>
#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Ports/LoadBalancerPort.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/FileUtils.h>


#include <iomanip>

using namespace std;
using namespace Uintah;

//__________________________________
//  ToDo
//  - copy On-the-Fly files directories
//______________________________________________________________________
//
UdaReducer::UdaReducer( const ProcessorGroup * world, 
                        const string         & udaDir ) :
  UintahParallelComponent( world ), d_udaDir(udaDir), d_dataArchive(0),
  d_timeIndex(0)
{
}

UdaReducer::~UdaReducer()
{
  delete d_dataArchive;
  
  for (unsigned int i = 0; i < d_savedLabels.size(); i++){
    VarLabel::destroy(d_savedLabels[i]);
  }
}
//______________________________________________________________________
//
void UdaReducer::problemSetup(const ProblemSpecP& prob_spec, 
                              const ProblemSpecP& restart_ps, 
                              GridP& grid, 
                              SimulationStateP& state)
{

  //__________________________________
  //  Add a warning message
  proc0cout << "\n______________________________________________________________________\n";
  proc0cout << "                      R E D U C E _ U D A \n\n";
  proc0cout << "    - If you're using this on a machine with a reduced set of system calls (mira) configure with\n";
  proc0cout << "          --with-boost\n";
  proc0cout << "      This will enable the non-system copy functions.\n\n";
  proc0cout << "    - You must manually copy all On-the-Fly files/directories from the original uda\n";
  proc0cout << "      to the new uda, reduce_uda ignores them.\n\n";
  proc0cout << "    - The <outputInterval>, <outputTimestepInterval> tags are ignored and every\n";
  proc0cout << "      timestep in the original uda is processed.  If you want to prune timesteps\n";
  proc0cout << "      you must manually delete timesteps directories and modify the index.xml file.\n\n";
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
  proc0cout << "    - ALWAYS, ALWAYS, ALWAYS verify that the new (modified) uda is consistent\n";
  proc0cout << "      with your specfications before deleting the original uda.\n\n";
  proc0cout << "______________________________________________________________________\n\n";
  
  //__________________________________
  //
  d_sharedState = state;
  d_sharedState->setIsLockstepAMR(true);
  d_sharedState->setSwitchState(true);         /// HACK NEED TO CHANGE THIS
   
  // This matl is for delT
  d_oneMatl = scinew SimpleMaterial();
  d_sharedState->registerSimpleMaterial( d_oneMatl );
  
  delt_label = d_sharedState->get_delt_label();

  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!d_dataArchiver){
    throw InternalError("reduceUda:couldn't get output port", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  Find timestep data from the original uda
  d_dataArchive = scinew DataArchive( d_udaDir, d_myworld->myrank(), d_myworld->size() );
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
    proc0cout << " *** Labels: " << varNames[i] << endl;
  }

  proc0cout << "\n";
}
//______________________________________________________________________
//
void UdaReducer::scheduleInitialize(const LevelP& level, 
                                    SchedulerP& sched)
{
  Task* t = scinew Task("UdaReducer::initialize", this, 
                       &UdaReducer::initialize);

  t->computes( delt_label, level.get_rep() );
  
  GridP grid = level->getGrid();
  d_lb = sched->getLoadBalancer();
  
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  t->setType(Task::OncePerProc);
  
  sched->addTask( t, perProcPatches, d_sharedState->allMaterials() );
}
//______________________________________________________________________
//
void UdaReducer::scheduleRestartInitialize(const LevelP& level,
                                           SchedulerP& sched)
{
}
//______________________________________________________________________
//  
void UdaReducer::initialize(const ProcessorGroup*,
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
void  UdaReducer::scheduleTimeAdvance( const LevelP& level, 
                                       SchedulerP& sched )
{
  sched_readDataArchive( level, sched );
}


//______________________________________________________________________
//    Schedule for each patch that this processor owns.
//    The DataArchiver::output() will cherry pick the variables to output

void UdaReducer::sched_readDataArchive(const LevelP& level,
                                      SchedulerP& sched){
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  const PatchSubset* patches = perProcPatches->getSubset(d_myworld->myrank());  
    
    
  Task* t = scinew Task("UdaReducer::readDataArchive", this, 
                        &UdaReducer::readDataArchive);
                                                  

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
    // Computerthe material set and subset 
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
void UdaReducer::readDataArchive(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{                           
  double time = d_times[d_timeIndex];
  int timestep = d_timesteps[d_timeIndex];
  proc0cout << "*** working on timestep: " << timestep << " physical time: " << time << endl;
  
  d_dataArchive->reduceUda_ReadUda(pg, d_timeIndex, d_oldGrid, patches, new_dw, d_lb);
  d_timeIndex++;
}


//______________________________________________________________________
//
void UdaReducer::scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP& sched)
{
  Task* t = scinew Task("UdaReducer::computeDelT",
                  this, &UdaReducer::computeDelT);
           
  t->computes( delt_label, level.get_rep() );
  
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  
  t->setType(Task::OncePerProc);
  sched->addTask( t, perProcPatches, d_sharedState->allMaterials() );
   
}

//______________________________________________________________________
//    This timestep is used to increment the physical time
void UdaReducer::computeDelT(const ProcessorGroup*,
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
double UdaReducer::getMaxTime()
{
  if (d_times.size() <= 1){
    return 0;
  }else {
    return d_times[d_times.size()-2]; // the last one is the hacked one, see problemSetup
  }
}
//______________________________________________________________________
//    Returns the physical time of the first output
double UdaReducer::getInitialTime()
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
UdaReducer::needRecompile( const double   /* time */,
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
GridP UdaReducer::getGrid() 
{ 
  GridP newGrid = d_dataArchive->queryGrid(d_timeIndex);
  
  if (d_oldGrid == nullptr || !(*newGrid.get_rep() == *d_oldGrid.get_rep())) {
    d_gridChanged = true;
    d_oldGrid = newGrid;
    d_lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return d_oldGrid; 
}




