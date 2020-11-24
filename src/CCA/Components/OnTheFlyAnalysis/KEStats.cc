/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/KEStats.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "KEStats:+" 
static DebugStream cout_doing("KEStats", false);
static DebugStream cout_dbg("KEStats", false);
//______________________________________________________________________        
KEStats::KEStats(const ProcessorGroup* myworld,
                                 const MaterialManagerP materialManager,
                                 const ProblemSpecP& module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set = 0;
  ps_lb = scinew KEStatsLabel();
}

//__________________________________
KEStats::~KEStats()
{
  cout_doing << " Doing: destorying KEStats " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  VarLabel::destroy(ps_lb->meanKELabel);
//  VarLabel::destroy(ps_lb->filePointerLabel);
//  VarLabel::destroy(ps_lb->filePointerLabel_preReloc);
  delete ps_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void KEStats::problemSetup(const ProblemSpecP& ,
                                   const ProblemSpecP& ,
                                   GridP& grid,
                                   std::vector<std::vector<const VarLabel* > > &PState,
                                   std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tKEStats" << endl;

//  d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  
  vector<int> m;
  m.push_back( 0 );
  
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());
  
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();   
  
  ps_lb->lastWriteTimeLabel =
    VarLabel::create("lastWriteTime_partE", max_vartype::getTypeDescription());

  ps_lb->meanKELabel =
    VarLabel::create("meanKE", max_vartype::getTypeDescription());
                                            
//  ps_lb->filePointerLabel =
//    VarLabel::create("filePointer", ParticleVariable< FILE* >::getTypeDescription() );

//  ps_lb->filePointerLabel_preReloc =
//    VarLabel::create("filePointer+", ParticleVariable< FILE* >::getTypeDescription() );
                                             
  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", m_analysisFreq);
  m_module_spec->require("numStepsAve",       d_numStepsAve);            

  //__________________________________
  //  Warning
  proc0cout << "\n\n__________________________________________________" << endl;
  proc0cout << "  WARNING      WARNING       WARNING" << endl;
  proc0cout << "     DataAnalysis:KEStats" << endl;
  proc0cout << "     <samplingFrequency> refers to 1.0/(time interval)" << endl;
  proc0cout << "     between computing stats on the Kinetic Energy" << endl;
  proc0cout << "________________________________________________\n\n" << endl;  
  
}

//______________________________________________________________________
void KEStats::scheduleInitialize(SchedulerP   & sched,
                                         const LevelP & level)
{

  printSchedule(level,cout_doing,"KEStats::scheduleInitialize");
  
  Task* t = scinew Task("KEStats::initialize", 
                  this, &KEStats::initialize);
  
  t->computes( ps_lb->lastWriteTimeLabel );
  t->computes( ps_lb->meanKELabel );
  sched->addTask( t, level->eachPatch(), d_matl_set );
}
//______________________________________________________________________
void KEStats::initialize(const ProcessorGroup *, 
                         const PatchSubset    * patches,
                         const MaterialSubset *,
                         DataWarehouse        *,
                         DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing KEStats::initialize");
     
    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put( max_vartype( tminus ), ps_lb->lastWriteTimeLabel );
    new_dw->put( max_vartype( 9.e9 ), ps_lb->meanKELabel );

    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:KEStats  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}
//______________________________________________________________________
void KEStats::scheduleRestartInitialize(SchedulerP   & sched,
                                                const LevelP & level)
{
  scheduleInitialize( sched, level);
}

//______________________________________________________________________
void KEStats::scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                                  const LevelP& level)
{ 
  printSchedule(level,cout_doing,"KEStats::scheduleDoAnalysis_preReloc");
  Task* t = scinew Task("KEStats::doAnalysis_preReloc", 
                   this,&KEStats::doAnalysis_preReloc);

  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.  Put it here so it will be registered during a restart
  sched->overrideVariableBehavior("filePointer", false, false, false, true, true);
                     
//  Ghost::GhostType gn = Ghost::None;
//  t->requires( Task::OldDW,  ps_lb->filePointerLabel, gn, 0 );
//  t->computes( ps_lb->filePointerLabel_preReloc  );
  
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}
//______________________________________________________________________
//
void KEStats::doAnalysis_preReloc(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch,cout_doing,"Doing KEStats::doAnalysis_preReloc");
    
#if 0
    int indx = d_matl->getDWIndex();
    
    ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
    constParticleVariable<FILE*> myFiles;
    ParticleVariable<FILE*>      myFiles_preReloc;

    new_dw->allocateAndPut( myFiles_preReloc, ps_lb->filePointerLabel_preReloc, pset );


    // Only transfer forward myFiles if they exist.  The filePointerLabel is NOT
    // saved in the checkpoints and so you can't get it from the old_dw.
    if( old_dw->exists( ps_lb->filePointerLabel, indx, patch ) ){
      old_dw->get( myFiles,  ps_lb->filePointerLabel,  pset );
      
      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        myFiles_preReloc[idx] = myFiles[idx];
      } 
    } else{
    
      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        myFiles_preReloc[idx] = nullptr;
      }
    }
#endif
  }
}   
 
//______________________________________________________________________
//
void KEStats::scheduleDoAnalysis(SchedulerP& sched,
                                 const LevelP& level)
{
  printSchedule(level,cout_doing,"KEStats::scheduleDoAnalysis");
  
  Task* t = scinew Task("KEStats::doAnalysis", 
                   this,&KEStats::doAnalysis);
                     
  sched_TimeVars( t, level, ps_lb->lastWriteTimeLabel, true );
  t->computes( ps_lb->meanKELabel );
  t->requires( Task::OldDW, ps_lb->meanKELabel );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void
KEStats::doAnalysis( const ProcessorGroup * pg,
                     const PatchSubset    * patches,
                     const MaterialSubset *,
                     DataWarehouse        * old_dw,
                     DataWarehouse        * new_dw )
{
  const Level* level = getLevel(patches);

  timeVars tv;

  max_vartype oldMeanKE;
  getTimeVars( old_dw, level, ps_lb->lastWriteTimeLabel, tv );
  putTimeVars( new_dw,        ps_lb->lastWriteTimeLabel, tv );
  old_dw->get( oldMeanKE,     ps_lb->meanKELabel);
  
  if( tv.isItTime == false ){
    new_dw->put( max_vartype( oldMeanKE ), ps_lb->meanKELabel );
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    int proc = m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);

    cout_dbg << Parallel::getMPIRank() << "   working on patch " 
             << patch->getID()         << " which is on proc " << proc << endl;
    //__________________________________
    // do analysis if this processor owns this patch and if it's time to do it
    if( proc == pg->myRank() ){
      // create the directory structure
      string udaDir = m_output->getOutputLocation();
      string path = udaDir + "/KineticEnergy.dat";

      // open the file
      ifstream KEfile(path.c_str());
      if(!KEfile){
        cerr << "KineticEnergy.dat file not opened, exiting" << endl;
        exit(1);
      }

      double time, KE;
      int numLines=0;
      double meanKE=0;
      while(KEfile >> time >> KE){
        meanKE+=KE;
        numLines++;
      }
      meanKE/=((double) numLines-1);
      cout << "meanKE = " << meanKE << endl;
      cout << "numLines = " << numLines << endl;
      new_dw->put( max_vartype( meanKE ), ps_lb->meanKELabel );
    }  // proc==pg...
  }  // patches
}
