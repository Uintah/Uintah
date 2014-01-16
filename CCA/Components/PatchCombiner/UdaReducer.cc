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
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>


#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/MaterialSetP.h>
#include <Core/Grid/Variables/VarTypes.h>


#include <iomanip>

using namespace std;
using namespace Uintah;

UdaReducer::UdaReducer(const ProcessorGroup* world, string udaDir)
  : UintahParallelComponent(world), d_udaDir(udaDir), d_dataArchive(0),
    d_timeIndex(0)
{
  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());
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
  for (unsigned int i = 0; i < d_labels.size(); i++){
    VarLabel::destroy(d_labels[i]);
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
  d_allMatlSet = d_sharedState->allMaterials(); 
  d_allMatlSubset = d_allMatlSet->getUnion();
  
  d_dataArchive = scinew DataArchive(d_udaDir, d_myworld->myrank(), d_myworld->size());
  d_dataArchive->queryTimesteps( d_timesteps, d_times );
  d_dataArchive->turnOffXMLCaching();

  // try to add a time to d_times that will get us passed the end of the
  // simulation -- yes this is a hack!!
  d_times.push_back(10 * d_times[d_times.size() - 1]);
  d_timesteps.push_back(999999999);

  //d_oldGrid = d_dataArchive->queryGrid(d_times[0]);
}
//______________________________________________________________________
//
void UdaReducer::scheduleInitialize(const LevelP& level, 
                                    SchedulerP& sched)
{
  lb = sched->getLoadBalancer();

  vector<string> varNames;
  vector< const TypeDescription *> typeDescriptions;
  
  d_dataArchive->queryVariables( varNames, typeDescriptions );
  
  for (unsigned int i = 0; i < varNames.size(); i++) {
    d_labels.push_back( VarLabel::create( varNames[i], typeDescriptions[i] ) );
  }
  
  //__________________________________
  //
  Task* t = scinew Task("UdaReducer::initialize", this, 
                         &UdaReducer::initialize);
   
//  const MaterialSet* all_matls = d_sharedState->allMaterials();
//  const MaterialSubset* all_matls_MS = all_matls->getUnion();

  t->computes( delt_label, level.get_rep() );
  sched->addTask(t, level->eachPatch(), d_allMatlSet);
}

//______________________________________________________________________
//
void UdaReducer::initialize(const ProcessorGroup*,
			       const PatchSubset* patches,
			       const MaterialSubset* /*matls*/,
			       DataWarehouse* /*old_dw*/,
			       DataWarehouse* new_dw)
{
  double t = d_times[0];
  delt_vartype delt_var = t; /* should subtract off start time -- this assumes it's 0 */
  new_dw->put(delt_var, delt_label);  
}



//______________________________________________________________________
//
void
UdaReducer::scheduleTimeAdvance( const LevelP& level, 
                                 SchedulerP& sched )
{

  sched_readAndSetVars( level, sched );

  //__________________________________
  //  Advance delT
  Task* t = scinew Task("UdaReducer::computeDelT", this, 
                         &UdaReducer::computeDelT );
                         
  t->computes( delt_label, level.get_rep() );
  
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = lb->getPerProcessorPatchSet(grid);
  
  sched->addTask( t, perProcPatches,  d_allMatlSet );
}


//______________________________________________________________________
//
void UdaReducer::sched_readAndSetVars(const LevelP& level,
                                      SchedulerP& sched){

  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = lb->getPerProcessorPatchSet(grid);

  // so we can tell the task which matls to use (sharedState doesn't have
  // any matls defined, so we can't use that).
  MaterialSetP allMatls = scinew MaterialSet();
  allMatls->createEmptySubsets(1);
  MaterialSubset* allMS = allMatls->getSubset(0);

  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;
  
  
  Task* t = scinew Task("UdaReducer::readAndSetVars", this, 
                        &UdaReducer::readAndSetVars);
  
  
  for (unsigned int i = 0; i < d_labels.size(); i++) {
    VarLabel* label = d_labels[i];

    ConsecutiveRangeSet matlsRangeSet;
    for (int i = 0; i < perProcPatches->getSubset(d_myworld->myrank())->size(); i++) {
      const Patch* patch = perProcPatches->getSubset(d_myworld->myrank())->get(i);
      matlsRangeSet = matlsRangeSet.unioned(d_dataArchive->queryMaterials(label->getName(), patch, d_timeIndex));
    }
    
    MaterialSet* matls;;
    if (prevMatlSet != 0 && prevRangeSet == matlsRangeSet) {
      matls = prevMatlSet.get_rep();
    }else {
     
      matls = scinew MaterialSet();
      vector<int> matls_vec;
      matls_vec.reserve(matlsRangeSet.size());
      
      for (ConsecutiveRangeSet::iterator iter = matlsRangeSet.begin(); iter != matlsRangeSet.end(); iter++) {
        if (!allMS->contains(*iter)) {
          allMS->add(*iter);
        }
	 matls_vec.push_back(*iter);
      }
      
      matls->addAll(matls_vec);
      prevRangeSet = matlsRangeSet;
      prevMatlSet  = matls;
    }

    allMS->sort();

    for (int l = 0; l < grid->numLevels(); l++) {
      t->computes(label, grid->getLevel(l)->allPatches()->getUnion(), matls->getUnion());
    }
  }

  //cout << d_myworld->myrank() << "  Done Calling DA::QuearyMaterials a bunch of times " << endl;

  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls.get_rep());

}


//______________________________________________________________________
//
void UdaReducer::readAndSetVars(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* /*old_dw*/,
                                DataWarehouse* new_dw)
{

  for(int p=0;p<patches->size();p++){ 
    const Patch* patch = patches->get(p);
    //__________________________________
    //output material indices
    if(patch->getID() == 0){

      cout << "//__________________________________Material Names:";
      int numAllMatls = d_sharedState->getNumMatls();
      cout << " numAllMatls " << numAllMatls << endl;
      for (int m = 0; m < numAllMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        cout <<" " << matl->getDWIndex() << ") " << matl->getName();
      }
      cout << "\n";
    }
  }


  double time = d_times[d_timeIndex];
  //int timestep = d_timesteps[d_timeIndex];

  if (d_timeIndex >= (int) (d_times.size()-1)) {
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
  
  proc0cout << "   Incrementing time " << d_timeIndex << " and time " << time << endl;


  d_dataArchive->restartInitialize(d_timeIndex, d_oldGrid, new_dw, lb, &time);
  d_timeIndex++;
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
//  If the number of materials on a level >

bool UdaReducer::needRecompile(double time, 
                               double dt,
                               const GridP& grid)
{
  bool recompile = d_gridChanged;
  d_gridChanged = false;

  vector<int> level_numMatls(d_oldGrid->numLevels());
  
  for (int L = 0; L < d_oldGrid->numLevels(); L++) {
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
    proc0cout << "     NEW GRID!!!!\n";
    d_oldGrid = newGrid;
    lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return d_oldGrid; 
}
