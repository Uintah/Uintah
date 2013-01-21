/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/1stLawThermo.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>

#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>

#include <Core/Util/DebugStream.h>

#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace Uintah;
using namespace std;
//______________________________________________________________________ 
//  To turn on the output
//  setenv SCI_DEBUG "FirstLawThermo_DBG_COUT:+" 
static DebugStream cout_doing("FirstLawThermo",   false);
static DebugStream cout_dbg("FirstLawThermo_dbg", false);
//______________________________________________________________________              
FirstLawThermo::FirstLawThermo(ProblemSpecP& module_spec,
                               SimulationStateP& sharedState,
                               Output* dataArchiver)
                               
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_zeroMatl = 0;
  d_matlSet  = 0;
  d_zeroPatch = 0;
  
  FL_lb = scinew FL_Labels();
  M_lb  = scinew MPMLabel();
  
  FL_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());

  FL_lb->fileVarsStructLabel=  VarLabel::create("FileInfo", 
                                            PerPatch<FileInfoP>::getTypeDescription());
}

//__________________________________
FirstLawThermo::~FirstLawThermo()
{
  cout_doing << " Doing: destorying FirstLawThermo " << endl;
  if( d_matlSet  && d_matlSet->removeReference() ) {
    delete d_matlSet;
  }
  if( d_zeroMatl && d_zeroMatl->removeReference() ) {
    delete d_zeroMatl;
  }
  if(d_zeroPatch && d_zeroPatch->removeReference())
    delete d_zeroPatch;
  
  VarLabel::destroy( FL_lb->lastWriteTimeLabel );
  VarLabel::destroy( FL_lb->fileVarsStructLabel );
  delete FL_lb;
  delete M_lb;
  
  // delete each plane
  vector<cv_face*>::iterator iter;
  for( iter  = d_cv_faces.begin();iter != d_cv_faces.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void FirstLawThermo::problemSetup(const ProblemSpecP& prob_spec,
                                  GridP& grid,
                                  SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tFirstLawThermo" << endl;
  
  if(!d_dataArchiver){
    throw InternalError("FirstLawThermo:couldn't get output port", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  Read in timing information
  d_prob_spec->require( "samplingFrequency", d_writeFreq );
  d_prob_spec->require( "timeStart",         d_StartTime );            
  d_prob_spec->require( "timeStop",          d_StopTime );
  
  // determine which material index to compute
  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  vector<int> m;
  m.push_back(0);                                 // needed for fileInfo
  m.push_back( d_matl->getDWIndex() );
  
  d_matlSet = scinew MaterialSet();
  d_matlSet->addAll_unique(m);                   // elimiate duplicate entries
  d_matlSet->addReference();
  d_matl_sub = d_matlSet->getUnion();
  
  // for fileInfo variable
  d_zeroMatl = scinew MaterialSubset();
  d_zeroMatl->add(0);
  d_zeroMatl->addReference();

  // one patch
  const Patch* p = grid->getPatchByID(0,0);
  d_zeroPatch = scinew PatchSet();
  d_zeroPatch->add(p);
  d_zeroPatch->addReference(); 
  
  //__________________________________
  // Loop over each face and find the extents
  ProblemSpecP cv_ps = prob_spec->findBlock("controlVolume");
   
  for (Patch::FaceType f = Patch::startFace; f <= Patch::endFace; f = Patch::nextFace(f)) {
  
    string facename = p->getFaceName(f);
    ProblemSpecP face_ps =cv_ps->findBlock(facename);
    
    map<string,string> faceMap;
    face_ps->getAttributes(faceMap);
    if (faceMap["extents"] == "partial"){
  //    face_ps->get("loPt", lo);
  //    face_ps->get("hiPt", hi);
    }
  }
}

//______________________________________________________________________
void FirstLawThermo::scheduleInitialize(SchedulerP& sched,
                                        const LevelP& level)
{
  printSchedule(level,cout_doing,"FirstLawThermo::scheduleInitialize");
  
  Task* t = scinew Task("FirstLawThermo::initialize",
                  this, &FirstLawThermo::initialize);
  
  t->computes(FL_lb->lastWriteTimeLabel);
  t->computes(FL_lb->fileVarsStructLabel, d_zeroMatl); 
  sched->addTask(t, d_zeroPatch, d_matlSet);
}
//______________________________________________________________________
void FirstLawThermo::initialize(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialize");
    
    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), FL_lb->lastWriteTimeLabel);

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;
    
    new_dw->put(fileInfo,    FL_lb->fileVarsStructLabel, 0, patch);
    
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:FirstLawThermo  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void FirstLawThermo::restartInitialize()
{
}

//______________________________________________________________________
void FirstLawThermo::scheduleDoAnalysis(SchedulerP& sched,
                                        const LevelP& level)
{

  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo", false, false, false, true, true);
  
  //__________________________________  
  //  compute the contributions from the various sources of energy
  Ghost::GhostType gn = Ghost::None;
  printSchedule(level,cout_doing,"FirstLawThermo::scheduleDoAnalysis");
  Task* t0 = scinew Task("FirstLawThermo::computeContributions", 
                    this,&FirstLawThermo::computeContributions);

  t0->requires(Task::OldDW, FL_lb->lastWriteTimeLabel);
  
  sched->addTask(t0, level->eachPatch(), d_matlSet);


  //__________________________________
  //  output the contributions
  Task* t1 = scinew Task("FirstLawThermo::doAnalysis", 
                    this,&FirstLawThermo::doAnalysis);
                    
  t1->requires(Task::OldDW, FL_lb->lastWriteTimeLabel);
  t1->requires(Task::OldDW, FL_lb->fileVarsStructLabel, d_zeroMatl, gn, 0);
  
  t1->computes(FL_lb->lastWriteTimeLabel);
  t1->computes(FL_lb->fileVarsStructLabel, d_zeroMatl);
  sched->addTask(t1, d_zeroPatch, d_matlSet);
}


//______________________________________________________________________
// 
void FirstLawThermo::computeContributions(const ProcessorGroup* pg,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matl_sub ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  max_vartype writeTime;
  old_dw->get(writeTime, FL_lb->lastWriteTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  if( now >= nextWriteTime ){

    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      printTask(patches, patch,cout_doing,"Doing computeContributions");
      
      
      
    }
  }
}



//______________________________________________________________________
// 
void FirstLawThermo::doAnalysis(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* matls ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
for(int p=0;p<patches->size();p++){
 const Patch* patch = patches->get(p);
 printTask(patches, patch,cout_doing,"Doing doAnalysis");
 }
  max_vartype writeTime;
  old_dw->get(writeTime, FL_lb->lastWriteTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  if( now >= nextWriteTime ){
    sum_vartype totalIntEng;
    sum_vartype totalXX;
    
    lastWriteTime = now;     
  }
  new_dw->put(max_vartype(lastWriteTime), FL_lb->lastWriteTimeLabel);    
}
