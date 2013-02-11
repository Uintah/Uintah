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

#include <CCA/Components/OnTheFlyAnalysis/MinMax.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>

#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StaticArray.h>
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
//  setenv SCI_DEBUG "MinMax_DBG_COUT:+" 
static DebugStream cout_doing("MinMax_DOING_COUT", false);
static DebugStream cout_dbg("MinMax_DBG_COUT", false);
//______________________________________________________________________    
/*  TO DO:
       - Only do a reduction every so often
       - 
______________________________________________________________________*/
          
MinMax::MinMax(ProblemSpecP& module_spec,
               SimulationStateP& sharedState,
               Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState = sharedState;
  d_prob_spec = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  d_zero_matl = 0;
  d_lb = scinew MinMaxLabel();
}

//__________________________________
MinMax::~MinMax()
{
  cout_doing << " Doing: destorying MinMax " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
   if(d_zero_matl && d_zero_matl->removeReference()) {
    delete d_zero_matl;
  } 
  
  VarLabel::destroy(d_lb->lastCompTimeLabel);
  VarLabel::destroy(d_lb->fileVarsStructLabel);
  delete d_lb;

}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void MinMax::problemSetup(const ProblemSpecP& prob_spec,
                          GridP& grid,
                          SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tMinMax" << endl;
  
  int numMatls  = d_sharedState->getNumMatls();
  if(!d_dataArchiver){
    throw InternalError("MinMax:couldn't get output port", __FILE__, __LINE__);
  }
                               
  d_lb->lastCompTimeLabel =  VarLabel::create("lastCompTime_minMax", 
                                            max_vartype::getTypeDescription());

  d_lb->fileVarsStructLabel = VarLabel::create("FileInfo_minMax", 
                                            PerPatch<FileInfoP>::getTypeDescription());       
                                            
  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_StartTime);            
  d_prob_spec->require("timeStop",          d_StopTime);

  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("MinMax: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 

  
  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  if(d_prob_spec->findBlock("material") ){
    d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    d_matl = d_sharedState->getMaterial(indx);
  } else {
    d_matl = d_sharedState->getMaterial(0);
  }
  
  int defaultMatl = d_matl->getDWIndex();
  
  //__________________________________
  //  Read in the optional material index from the variables that may be different
  //  from the default index
  vector<int> m;
  
  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);
  d_matl_set = scinew MaterialSet();
  map<string,string> attribute;
    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
   
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }
    
    // bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("MinMax: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    
    d_varMatl.push_back(matl);
    m.push_back(matl);
  }
  
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  //Construct the matl_set
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // for fileInfo variable
  d_zero_matl = scinew MaterialSubset();
  d_zero_matl->add(0);
  d_zero_matl->addReference();
  
  //__________________________________
  //  Read in variables label names                
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
    
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("MinMax: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    //__________________________________
    //  Bulletproofing
    // The user must specify the matl for single matl variables
    if ( name == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("MinMax: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
    }
    
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();
    
    //__________________________________
    bool throwException = false;  
    
    // only CC, SFCX, SFCY, SFCZ variables
    if(td->getType() != TypeDescription::CCVariable &&
       td->getType() != TypeDescription::SFCXVariable &&
       td->getType() != TypeDescription::SFCYVariable &&
       td->getType() != TypeDescription::SFCZVariable ){
       throwException = true;
    }
    // CC Variables, only Doubles and Vectors 
    if(td->getType() != TypeDescription::CCVariable &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::int_type &&
       subtype->getType() != TypeDescription::Vector  ){
      throwException = true;
    }
    // Face Centered Vars, only Doubles
    if( (td->getType() == TypeDescription::SFCXVariable ||
         td->getType() == TypeDescription::SFCYVariable ||
         td->getType() == TypeDescription::SFCZVariable) &&
         subtype->getType() != TypeDescription::double_type) {
      throwException = true;
    } 
    if(throwException){       
      ostringstream warn;
      warn << "ERROR:AnalysisModule:MinMax: ("<<label->getName() << " " 
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    d_varLabels.push_back(label);
  }
}

//______________________________________________________________________
void MinMax::scheduleInitialize(SchedulerP& sched,
                                const LevelP& level)
{
  printSchedule(level,cout_doing,"minMax::scheduleInitialize");
  Task* t = scinew Task("MinMax::initialize", 
                  this, &MinMax::initialize);
  
  t->computes(d_lb->lastCompTimeLabel);
  t->computes(d_lb->fileVarsStructLabel, d_zero_matl); 
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void MinMax::initialize(const ProcessorGroup*, 
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  cout_doing << "Doing MinMax:Initialize" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing MinMax::initialize");
    
    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), d_lb->lastCompTimeLabel);

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;
    
    new_dw->put(fileInfo,    d_lb->fileVarsStructLabel, 0, patch);
    
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:MinMax  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void MinMax::restartInitialize()
{
// need to do something here
//  new_dw->put(max_vartype(0.0), d_lb->lastCompTimeLabel);
}

//______________________________________________________________________
void MinMax::scheduleDoAnalysis(SchedulerP& sched,
                                const LevelP& level)
{
  printSchedule(level,cout_doing,"MinMax::scheduleDoAnalysis");
   
  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_minMax", false, false, false, true, true); 
  
  Ghost::GhostType gn = Ghost::None;
  
  //__________________________________
  //  computeMinMax task;     
  Task* t0 = scinew Task( "MinMax::computeMinMax", 
                     this,&MinMax::computeMinMax );           
  t0->requires( Task::OldDW, d_lb->lastCompTimeLabel );
 
  for ( unsigned int i =0 ; i < d_varLabels.size(); i++ ) {
    // bulletproofing
    if( d_varLabels[i] == NULL ){
      string name = d_varLabels[i]->getName();
      throw InternalError("MinMax: scheduleDoAnalysis label not found: " 
                          + name , __FILE__, __LINE__);
    }
    
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add(d_varMatl[i]);
    matSubSet->addReference();
    
    t0->requires( Task::NewDW,d_varLabels[i], matSubSet, gn, 0 );
    
    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }
  sched->addTask( t0, level->eachPatch(), d_matl_set );
 
  
  //__________________________________
  //  Write write min/max to a  file

  Task* t1 = scinew Task( "MinMax::doAnalysis", 
                       this,&MinMax::doAnalysis );      
                            
  t1->requires( Task::OldDW, d_lb->lastCompTimeLabel );
  t1->requires( Task::OldDW, d_lb->fileVarsStructLabel, d_zero_matl, gn, 0 );
  
  t1->computes( d_lb->lastCompTimeLabel );
  t1->computes( d_lb->fileVarsStructLabel, d_zero_matl );
  
  sched->addTask( t1, level->eachPatch(), d_matl_set );
}

//______________________________________________________________________
//  This task computes and min/max of each variable
//
void MinMax::computeMinMax(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{    
  const Level* level = getLevel(patches);
  
  max_vartype writeTime;
  old_dw->get(writeTime, d_lb->lastCompTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);    
    
    cout << " HERE " << endl;
    
    //__________________________________
    // compute min/max if it's time to write
    if( now >= nextWriteTime){
    
      printTask(patches, patch,cout_doing,"Doing MinMax::computeMinMax");
            
      Ghost::GhostType gn = Ghost::None;

      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
        string name = d_varLabels[i]->getName();
        // bulletproofing
        if(d_varLabels[i] == NULL){
          throw InternalError("MinMax: analyze label not found: " 
                          + name , __FILE__, __LINE__);
        }

        const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
        const Uintah::TypeDescription* subtype = td->getSubType();


        int indx = d_varMatl[i];
        
        IntVector maxIndx( -SHRT_MAX, -SHRT_MAX, -SHRT_MAX);
        IntVector minIndx(  SHRT_MAX,  SHRT_MAX,  SHRT_MAX);
        Point maxPos(-DBL_MAX,-DBL_MAX,-DBL_MAX);
        Point minPos( DBL_MAX, DBL_MAX, DBL_MAX);
        
        switch(td->getType()){
          case Uintah::TypeDescription::CCVariable:      // CC Variables
            switch(subtype->getType()) {
            
            case Uintah::TypeDescription::double_type:{
              constCCVariable<double> q_CC;
              new_dw->get(q_CC, d_varLabels[i], indx, patch, gn, 0);
              double max = -DBL_MAX;
              double min = DBL_MAX;
              
              
              for (CellIterator iter=patch->getCellIterator();!iter.done();iter++) {
                IntVector c = *iter;
                double Q = q_CC[c];
                if ( Q >= max ){
                  max = Q;
                  maxIndx = c;
                }
                
                if (Q <= min ){
                  min = Q;
                  minIndx = c;
                }
              }    
              Point maxPos = level->getCellPosition(maxIndx);
              Point minPos = level->getCellPosition(minIndx);          
              
              cout << name << " max: " << max << " " << maxIndx << " maxPos " << maxPos << endl;
              cout << "         min: " << min << " " << minIndx << " minPos " << minPos << endl;
              break;
            } 
            case Uintah::TypeDescription::Vector: {
              constCCVariable<Vector> q_CC;
              new_dw->get(q_CC, d_varLabels[i], indx, patch, gn, 0);
              break;
            }  
            case Uintah::TypeDescription::int_type: {
              constCCVariable<int> q_CC;
              new_dw->get(q_CC, d_varLabels[i], indx, patch, gn, 0);
              break; 
            }
            default:
              throw InternalError("MinMax: invalid data type", __FILE__, __LINE__); 
            }
            break;
          case Uintah::TypeDescription::SFCXVariable: {   // SFCX Variables
            constSFCXVariable<double> q_SFCX;
            new_dw->get( q_SFCX, d_varLabels[i], indx, patch, gn, 0 );
            break;
          }
          case Uintah::TypeDescription::SFCYVariable: {   // SFCY Variables
            constSFCYVariable<double> q_SFCY;  
            new_dw->get( q_SFCY, d_varLabels[i], indx, patch, gn, 0 );
            break;
          }
          case Uintah::TypeDescription::SFCZVariable: {   // SFCZ Variables
            constSFCZVariable<double> q_SFCZ;
            new_dw->get( q_SFCZ, d_varLabels[i], indx, patch, gn, 0 );
            break;
          }
          default:
            ostringstream warn;
            warn << "ERROR:AnalysisModule:lineExtact: ("<<d_varLabels[i]->getName() << " " 
                 << td->getName() << " ) has not been implemented" << endl;
            throw InternalError(warn.str(), __FILE__, __LINE__);
        }
        
      }            
         
    }  // time to write data
  }  // patches
}

//______________________________________________________________________
//  This task writes out the min/max of each var label to a separate
//  file.
void MinMax::doAnalysis(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  UintahParallelComponent* DA = dynamic_cast<UintahParallelComponent*>(d_dataArchiver);
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>( DA->getPort("load balancer"));
    
  const Level* level = getLevel(patches);
  Ghost::GhostType gn = Ghost::None;
    
  max_vartype writeTime;
  old_dw->get( writeTime, d_lb->lastCompTimeLabel );
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    // open the struct that contains a map of the file pointers 
    // Note: after regridding this may not exist for this patch in the old_dw
    PerPatch<FileInfoP> fileInfo;
    
    if( old_dw->exists( d_lb->fileVarsStructLabel, 0, patch ) ){
      old_dw->get( fileInfo, d_lb->fileVarsStructLabel, 0, patch );
    }else{  
      FileInfo* myFileInfo = scinew FileInfo();
      fileInfo.get() = myFileInfo;
    }
    
    std::map<string, FILE *> myFiles;

    if( fileInfo.get().get_rep() ){
      myFiles = fileInfo.get().get_rep()->files;
    }    
    
    int proc = lb->getPatchwiseProcessorAssignment(patch);
    cout_dbg << Parallel::getMPIRank() << "   working on patch " << patch->getID() << " which is on proc " << proc << endl;
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myrank() && now >= nextWriteTime){  

#if 1 
      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
        string name = d_varLabels[i]->getName();
         
        // bulletproofing
        if(d_varLabels[i] == NULL){
          
          throw InternalError("MinMax: analyze label not found: " 
                          + name , __FILE__, __LINE__);
        }
            
      
        //__________________________________
        // create the directory structure
        string udaDir = d_dataArchiver->getOutputLocation();

        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        string path = udaDir + "/" + levelIndex;

        if( d_isDirCreated.count(path) == 0){
          createDirectory(path, levelIndex);
          d_isDirCreated.insert(path);
        }
        
        ostringstream fname;
        fname<<path<<"/"<<name <<"("<<d_varMatl[i]<<")";
        string filename = fname.str();

        //__________________________________
        //  Open the file pointer 
        //  if it's not in the fileInfo struct then create it
        FILE *fp;

        if( myFiles.count(filename) == 0 ){
          createFile(filename, fp);
          myFiles[filename] = fp;

        } else {
          fp = myFiles[filename];
        }

        if (!fp){
          throw InternalError("\nERROR:dataAnalysisModule:MinMax:  failed opening file"+filename,__FILE__, __LINE__);
        }   
#endif    
      }    
      lastWriteTime = now;     
    }  // time to write data
    
    // put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents   
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo,                   d_lb->fileVarsStructLabel, 0, patch);
    new_dw->put(max_vartype(lastWriteTime), d_lb->lastCompTimeLabel); 
  }  // patches
}



//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void MinMax::createFile(string& filename,  FILE*& fp)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }
  
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"Time    max    min\n");
  
  cout << Parallel::getMPIRank() << " MinMax:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
//
void
MinMax::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << "MinMax:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  } else {
    closedir(check);
  }
  
  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL ) {
    cout << "MinMax:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
