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




//______________________________________________________________________
//
UdaReducer::UdaReducer(const ProcessorGroup* world, 
                       string udaDir)
  : UintahParallelComponent(world), d_udaDir(udaDir), d_dataArchive(0),
    d_timeIndex(0)
{
//  delt_label = VarLabel::create("delT", delt_vartype::getTypeDescription());
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
    throw InternalError("ICE:couldn't get output port", __FILE__, __LINE__);
  }

  
  //__________________________________
  //  Find out the time data for the uda
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
    cout << " label: " << varNames[i] << endl;
  }
  cout << " delT: " << delt_label->getName() << endl; 
}
//______________________________________________________________________
//
void UdaReducer::scheduleInitialize(const LevelP& level, 
                                    SchedulerP& sched)
{

  d_lb = sched->getLoadBalancer();
  


#if 0
  Task* t = scinew Task("UdaReducer::initialize", this, 
                       &UdaReducer::initialize);

  t->computes( delt_label, level.get_rep() );
  
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  t->setType(Task::OncePerProc);
  
  sched->addTask( t, perProcPatches, d_sharedState->allMaterials() );
#endif
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
//  This task is only called once.
void
UdaReducer::scheduleTimeAdvance( const LevelP& level, 
                                 SchedulerP& sched )
{

  sched_readAndSetVars( level, sched );

#if 0
  //__________________________________
  //  Advance delT
  Task* t = scinew Task("UdaReducer::computeDelT", this, 
                         &UdaReducer::computeDelT );
                         
  t->computes( delt_label, level.get_rep() );
  
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  
  sched->addTask( t, perProcPatches,  d_allMatlSet );
#endif
}


//______________________________________________________________________
//
void UdaReducer::sched_readAndSetVars(const LevelP& level,
                                      SchedulerP& sched){
  cout <<"__________________________________sched_readAndSetVars: " << endl;
  GridP grid = level->getGrid();
  const PatchSet* perProcPatches = d_lb->getPerProcessorPatchSet(grid);
  const PatchSubset* patches = perProcPatches->getSubset(d_myworld->myrank());  
    
    
  Task* t = scinew Task("UdaReducer::readAndSetVars", this, 
                       &UdaReducer::readAndSetVars);
                                                  

///__________________________________
//  Attempt 3
#if 0
  std::vector< Output::SavedLabels2 > savedLabels;
  savedLabels = d_dataArchiver->getSavedLabels();
  
  cout << "  savedLabels.size: " << savedLabels.size() << endl;
  
  vector< Output::SavedLabels2 >::iterator iter;

  for(iter = savedLabels.begin(); iter!= savedLabels.end(); iter++) { 
    const Output::SavedLabels2 me = *iter;
    
  
    cout <<"  " << me.labelName  << " matls " << me.matls << " levels: " << me.levels <<endl;
    VarLabel* var = VarLabel::find( me.labelName);
    
    if (var == NULL) {
      throw ProblemSetupException( me.labelName +" variable not found to save.", __FILE__, __LINE__);
    }
    
    MaterialSet* matlSet = scinew MaterialSet();
    // loop over levels
    

    for (ConsecutiveRangeSet::iterator level = me.levels.begin(); level != me.levels.end(); level++) {
      cout << "  Levels: " << *level << endl;
      
      vector<int> m;
      
      if( me.matls != ConsecutiveRangeSet::all ){
        for (ConsecutiveRangeSet::iterator iter = me.matls.begin(); iter != me.matls.end(); iter++) {
          m.push_back(*iter);
        }
      }
      
      matlSet->addAll(m);
    }
    cout << "  matlSet: " << *matlSet << endl;
  } 
#endif  
#if 0 
//______________________________________________________________________
//  Attempt 2
  vector< Output::SavedLabels >::iterator saveIter;
  //const PatchSet* patches = lb->getOutputPerProcessorPatchSet(level);

  for(saveIter = savedLabels.begin(); saveIter!= savedLabels.end(); saveIter++) {
    // check to see if the input file requested to save on this level.
    // check is done by absolute level, or relative to end of levels (-1 finest, -2 second finest,...)
    
    const VarLabel* label = (*saveIter).varLabel;
    map<int, MaterialSetP>::iterator matlSet;
  
    matlSet = saveIter->matlSet.find( level->getIndex() );
    
    cout <<"  readAndSetVars: " << label->getName() << " MatlSet: " << matlSet->second.get_rep() << endl;
    
#if 0    
    if ( matlSet == saveIter->matlSet.end() ){
      matlSet = saveIter->matlSet.find(level->getIndex() - level->getGrid()->numLevels());
    }
    if ( matlSet == saveIter->matlSet.end() ){
      matlSet = saveIter->matlSet.find(ALL_LEVELS);
    }
#endif

    if ( matlSet != saveIter->matlSet.end() ) {

      const MaterialSubset* matlSubset = matlSet->second.get_rep()->getUnion();

      // out of domain really is only there to handle the "all-in-one material", but doesn't break anything else
      t->computes( label, matlSubset );
      cout <<" requires: " << label->getName() << " MatlSubset: " << *matlSubset << endl;
    }
  }
  
  
  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, d_sharedState->allMaterials() ); 
#endif  
  
    
  
  #if 1                 

  // manually determine with matls for scheduling.
  // The sharedState does not have any materials registered
  // so you have to do it manually
  MaterialSetP allMatls = scinew MaterialSet();
  allMatls->createEmptySubsets(1);
  MaterialSubset* allMatlSubset = allMatls->getSubset(0);

  MaterialSetP prevMatlSet = 0;
  ConsecutiveRangeSet prevRangeSet;
  
//  std::vector< Output::SavedLabels > DA_savedLabels;
//  DA_savedLabels = d_dataArchiver->getSavedLabels();  

  
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
    // Compute all labels on all levels.  The DataArchiver::output task
    // will then pick and choose which variables to write.
    for (int L = 0; L < grid->numLevels(); L++) {
      LevelP level = grid->getLevel(L);
      t->computes(label, level->allPatches()->getUnion(), matlSet->getUnion());
    }
  }  // loop savedLabels

  t->setType(Task::OncePerProc);
  sched->addTask(t, perProcPatches, allMatls.get_rep());
#endif

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
    proc0cout << "//__________________________________Material Names:";
    int numAllMatls = d_sharedState->getNumMatls();
    proc0cout << " numAllMatls " << numAllMatls << endl;
    
    for (int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      proc0cout <<" " << matl->getDWIndex() << ") " << matl->getName();
    }
    proc0cout << "\n";
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
  
  cout << "perProcPatches "<< *perProcPatches << endl;
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
    d_lb->possiblyDynamicallyReallocate(newGrid, true);
  }
  return d_oldGrid; 
}

//______________________________________________________________________
//
void UdaReducer::scheduleFinalizeTimestep(const LevelP& level, 
                                          SchedulerP&)
{
  cout << " UdaReducer::scheduleFinalizeTimestep " << endl;
}
