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

#include <CCA/Components/PatchCombiner/UdaReducer.h>
#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Ports/LoadBalancer.h>
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
//  - getPerProcessorePatchSet(level)
//  - add warning message.
//  Testing
//     - oututput double as float
//     - timestep numbers
//     - vary the output intervals
//     - particles on a single level
//     - On the Fly files/directories?
//______________________________________________________________________
//
UdaReducer::UdaReducer(const ProcessorGroup* world, 
                       string udaDir)
  : UintahParallelComponent(world), d_udaDir(udaDir), d_dataArchive(0),
    d_timeIndex(0)
{
}

UdaReducer::~UdaReducer()
{

  if(d_allMatlSet && d_allMatlSet->removeReference()) {
    delete d_allMatlSet;
  }
   if(d_allMatlSubset && d_allMatlSubset->removeReference()) {
    delete d_allMatlSubset;
  }
  delete d_dataArchive;
  
  VarLabel::destroy(delt_label);
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
  d_sharedState = state;
  d_sharedState->setIsLockstepAMR(true);
  d_sharedState->d_switchState = true;         /// HACK NEED TO CHANGE THIS
   
  // This matl is for delT
  d_oneMatl = scinew SimpleMaterial();
  d_sharedState->registerSimpleMaterial( d_oneMatl );
  
  delt_label = d_sharedState->get_delt_label();

  d_dataArchiver = dynamic_cast<Output*>(getPort("output"));
  if(!d_dataArchiver){
    throw InternalError("reduceUda:couldn't get output port", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  Find out the time data from the uda
  d_dataArchive = scinew DataArchive(d_udaDir, d_myworld->myrank(), d_myworld->size());
  d_dataArchive->queryTimesteps( d_timesteps, d_times );  
  d_dataArchive->turnOffXMLCaching();
  
  // try to add a time to d_times that will get us passed the end of the
  // simulation -- yes this is a hack!!
  d_times.push_back(10 * d_times[d_times.size() - 1]);
  d_timesteps.push_back(999999999);

  //__________________________________
  //  define the varLabels
  vector<string> varNames;
  vector< const TypeDescription *> typeDescriptions;
  d_dataArchive->queryVariables( varNames, typeDescriptions );
  
  for (unsigned int i = 0; i < varNames.size(); i++) {
    d_savedLabels.push_back( VarLabel::create( varNames[i], typeDescriptions[i] ) );
    proc0cout << " *** Labels: " << varNames[i] << endl;
  }
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
//  Set the timestep number and first delT
void UdaReducer::initialize(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* /*matls*/,
			       DataWarehouse* /*old_dw*/,
			       DataWarehouse* new_dw)
{
  double t = d_times[0];
  delt_vartype delt_var = t;
  new_dw->put( delt_var, delt_label );
}

//______________________________________________________________________
//  This task is only called once.
void  UdaReducer::scheduleTimeAdvance( const LevelP& level, 
                                       SchedulerP& sched )
{
  sched_computeLabels( level, sched );
}


//______________________________________________________________________
//
// This tasks determines the material subset for each label
// and add a computes().  You want to compute all of the varLabels on all
// levels.  The DataArchiver::output() will cherry pick the variables to output

void UdaReducer::sched_computeLabels(const LevelP& level,
                                      SchedulerP& sched){
  proc0cout <<"__________________________________sched_computeLabels: " << endl;
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  const PatchSubset* patches = perProcPatches->getSubset(d_myworld->myrank());  
    
    
  Task* t = scinew Task("UdaReducer::computeLabels", this, 
                       &UdaReducer::computeLabels);
                                                  

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
    if (prevMatlSet != 0 && prevRangeSet == matlsRangeSet) {
      matlSet = prevMatlSet.get_rep();
    } else {
     
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
    // Compute all of the labels on all levels.  The DataArchiver::output task
    // will then pick and choose which variables to write based on the input file
    for (int L = 0; L < grid->numLevels(); L++) {
      LevelP level = grid->getLevel(L);
      t->computes(label, level->allPatches()->getUnion(), matlSet->getUnion());
    }
  }  // loop savedLabels

  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls.get_rep());

}


//______________________________________________________________________
//  This task outputs diagnostic information every timestep.  Note
//  The DataArchiever::output() task handles the output
//
void UdaReducer::computeLabels(const ProcessorGroup*,
                                const PatchSubset*,
                                const MaterialSubset* matls,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{

  double time = d_times[d_timeIndex];
  int timestep = d_timesteps[d_timeIndex];

  if ( d_timeIndex >= (int) (d_times.size()-1) ) {
    // error situation - we have run out of timesteps in the uda, but 
    // the time does not satisfy the maxTime, so the simulation wants to 
    // keep going
    cerr << "The timesteps in the uda directory do not extend to the maxTime\n"
         << "in the input.ups file.  To not get this exception, adjust the\n"
         << "maxTime in <udadir>/input.xml to be\n"
         << "between " << (d_times.size() >= 3 ? d_times[d_times.size()-3] : 0)
         << " and " << d_times[d_times.size()-2] << " (the last time in the uda)\n"
         << "This is not a critical error - it just adds one more timestep\n"
         << "that you may have to remove manually\n\n";
  }
  
  proc0cout << "*** computeLabels Incrementing timeIndex " << d_timeIndex << " timestep: " << timestep << " time " << time << endl;


  d_dataArchive->restartInitialize(d_timeIndex, d_oldGrid, new_dw, d_lb, &time);
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
//
void UdaReducer::computeDelT(const ProcessorGroup*,
                             const PatchSubset*,
                             const MaterialSubset*,
                             DataWarehouse* /*old_dw*/,
                             DataWarehouse* new_dw)
  
{
  // don't use the delt produced in restartInitialize.
  double delt = d_times[d_timeIndex] - d_times[d_timeIndex-1];
  delt_vartype delt_var = delt;
  new_dw->put(delt_var, delt_label); 
  proc0cout << "*** computeDelT" << endl; 
}



//______________________________________________________________________
//
double UdaReducer::getMaxTime()
{
  if (d_times.size() <= 1){
    return 0;
  }else {
    return d_times[d_times.size()-2]; // the last one is the hacked one, see problemSetup
  }
}

//______________________________________________________________________
//  If the number of materials on a level changes or if the grid
//  has changed then call for a recompile

bool UdaReducer::needRecompile(double time, 
                               double dt,
                               const GridP& grid)
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
  return recompile;
}

//______________________________________________________________________
// called by the SimulationController once per timestep
GridP UdaReducer::getGrid() 
{ 
  GridP newGrid = d_dataArchive->queryGrid(d_timeIndex);
  
  if (d_oldGrid == 0 || !(*newGrid.get_rep() == *d_oldGrid.get_rep())) {
    d_gridChanged = true;
    d_oldGrid = newGrid;
    d_lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return d_oldGrid; 
}

//______________________________________________________________________
//
void UdaReducer::scheduleFinalizeTimestep(const LevelP& level, 
                                          SchedulerP& sched)
{
  Task* t = scinew Task("UdaReducer::finalizeTimestep",
                  this, &UdaReducer::finalizeTimestep);

  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  
  t->setType(Task::OncePerProc);
  sched->addTask( t, perProcPatches, d_sharedState->allMaterials() );
   
}

//______________________________________________________________________
//
void UdaReducer::finalizeTimestep(const ProcessorGroup*,
                                  const PatchSubset*,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse*)
  
{
  int index = d_timesteps[d_timeIndex];
  proc0cout << "*** finalizeTimestep" << endl;
  //proc0cout << "*** Now incrementing the timestep index " << index << endl;  
  //d_sharedState->setCurrentTopLevelTimeStep( index );
}





