/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/visit_defs.h>

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#define ALL_LEVELS 99
#define FINEST_LEVEL -1 
using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "MinMax_DBG_COUT:+" 
static DebugStream cout_doing("MinMax_DOING_COUT", "OnTheFlyAnalysis", "Min/Max debug stream", false);

//______________________________________________________________________    
/*  TO DO:
       - Find a way to keep track of the min/max points for each variable
       - Conditional scheduling of the tasks.  Currently, the reductions
         are occuring every timestep.
         
       
______________________________________________________________________*/
          
MinMax::MinMax( const ProcessorGroup* myworld,
                const MaterialManagerP materialManager,
                const ProblemSpecP& module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set = nullptr;
  d_zero_matl = nullptr;
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
  
  // d_analyzeVars 
  for ( unsigned int i =0 ; i < d_analyzeVars.size(); i++ ) {
    VarLabel::destroy( d_analyzeVars[i].reductionMinLabel );
    VarLabel::destroy( d_analyzeVars[i].reductionMaxLabel );
    
    if( d_analyzeVars[i].matSubSet && d_analyzeVars[i].matSubSet->removeReference()){
      delete d_analyzeVars[i].matSubSet;
    }
  }
  
  delete d_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void MinMax::problemSetup(const ProblemSpecP&,
                          const ProblemSpecP&,
                          GridP& grid,
                          std::vector<std::vector<const VarLabel* > > &PState,
                          std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tMinMax" << endl;
  
  int numMatls  = m_materialManager->getNumMatls();

  d_lb->lastCompTimeLabel =  VarLabel::create("lastCompTime_minMax", 
                                              max_vartype::getTypeDescription() );

  d_lb->fileVarsStructLabel = VarLabel::create("FileInfo_minMax", 
                                               PerPatch<FileInfoP>::getTypeDescription() );       
                                            
  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", m_analysisFreq);
  m_module_spec->require("timeStart",         d_startTime);            
  m_module_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("MinMax: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 
  
  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  if(m_module_spec->findBlock("material") ){
    d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  } else if (m_module_spec->findBlock("materialIndex") ){
    int indx;
    m_module_spec->get("materialIndex", indx);
    d_matl = m_materialManager->getMaterial(indx);
  } else {
    d_matl = m_materialManager->getMaterial(0);
  }
  
  int defaultMatl = d_matl->getDWIndex();
  
  //__________________________________
  vector<int> m;
  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);
  map<string,string> attribute;
    
  //__________________________________
  //  Now loop over all the variables to be analyzed  
    
  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {

    var_spec->getAttributes( attribute );
    
    //__________________________________
    // Read in the variable name
    string labelName = attribute["label"];
    VarLabel* label = VarLabel::find(labelName);
    if( label == nullptr ){
      throw ProblemSetupException("MinMax: analyze label not found: " + labelName , __FILE__, __LINE__);
    }
    
    // Bulletproofing - The user must specify the matl for single matl
    // variables
    if ( labelName == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("MinMax: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
    }

    // Read in the optional level index
    int level = ALL_LEVELS;
    if (attribute["level"].empty() == false){
      level = atoi(attribute["level"].c_str());
    }
    
    //  Read in the optional material index from the variables that
    //  may be different from the default index and construct the
    //  material set
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }
    
    // Bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("MinMax: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    
    m.push_back(matl);
    
    //__________________________________
    bool throwException = false;  
    
    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();
    
    const int baseType = td->getType();
    const int subType  = subtype->getType();
    
    // only CC, SFCX, SFCY, SFCZ variables
    if(baseType != TypeDescription::CCVariable &&
       baseType != TypeDescription::NCVariable &&
       baseType != TypeDescription::SFCXVariable &&
       baseType != TypeDescription::SFCYVariable &&
       baseType != TypeDescription::SFCZVariable ){
       throwException = true;
    }
    // CC Variables, only Doubles and Vectors 
    if(baseType != TypeDescription::CCVariable &&
       subType  != TypeDescription::double_type &&
       subType  != TypeDescription::Vector  ){
      throwException = true;
    }
    // NC Variables, only Doubles and Vectors 
    if(baseType != TypeDescription::NCVariable &&
       subType  != TypeDescription::double_type &&
       subType  != TypeDescription::Vector  ){
      throwException = true;
    }
    // Face Centered Vars, only Doubles
    if( (baseType == TypeDescription::SFCXVariable ||
         baseType == TypeDescription::SFCYVariable ||
         baseType == TypeDescription::SFCZVariable) &&
         subType != TypeDescription::double_type) {
      throwException = true;
    } 

    if(throwException){       
      ostringstream warn;
      warn << "ERROR:AnalysisModule:MinMax: ("<<label->getName() << " " 
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__); 
    }

    //__________________________________
    //  Create the min and max VarLabels for the reduction variables
    string VLmax = labelName + "_max";
    string VLmin = labelName + "_min";
    VarLabel* meMax = nullptr;
    VarLabel* meMin = nullptr;
    
    // double
    if( subType == TypeDescription::double_type ) {
      meMax = VarLabel::create( VLmax, max_vartype::getTypeDescription() );
      meMin = VarLabel::create( VLmin, min_vartype::getTypeDescription() );
    }
    // Vectors
    if( subType == TypeDescription::Vector ) {
      meMax = VarLabel::create( VLmax, maxvec_vartype::getTypeDescription() );
      meMin = VarLabel::create( VLmin, minvec_vartype::getTypeDescription() );
    }    

    varProperties me;
    me.label = label;
    me.matl  = matl;
    me.level = level;
    me.reductionMaxLabel = meMax;
    me.reductionMinLabel = meMin;
    me.matSubSet = scinew MaterialSubset();
    me.matSubSet->add( matl );
    me.matSubSet->addReference();
    
    d_analyzeVars.push_back(me);
  
#ifdef HAVE_VISIT
    static bool initialized = false;

    if( m_application->getVisIt() && !initialized ) {
      ApplicationInterface::analysisVar aVar;
      aVar.component = "Analysis-MinMax";
      aVar.name  = label->getName();
      aVar.matl  = matl;
      aVar.level = level;
      aVar.labels.push_back( meMin );
      aVar.labels.push_back( meMax );    
      m_application->getAnalysisVars().push_back(aVar);

      ApplicationInterface::interactiveVar var;
      var.component = "Analysis-MinMax";
      var.name       = "SamplingFrequency";
      var.type       = Uintah::TypeDescription::double_type;
      var.value      = (void *) &m_analysisFreq;
      var.range[0]   = 0;
      var.range[1]   = 1e99;
      var.modifiable = true;
      var.recompile  = false;
      var.modified   = false;
      m_application->getUPSVars().push_back( var );

      var.component = "Analysis-MinMax";
      var.name       = "StartTime";
      var.type       = Uintah::TypeDescription::double_type;
      var.value      = (void *) &d_startTime;
      var.range[0]   = 0;
      var.range[1]   = 1e99;
      var.modifiable = true;
      var.recompile  = false;
      var.modified   = false;
      m_application->getUPSVars().push_back( var );
      
      var.component = "Analysis-MinMax";
      var.name       = "StopTime";
      var.type       = Uintah::TypeDescription::double_type;
      var.value      = (void *) &d_stopTime;
      var.range[0]   = 0;
      var.range[1]   = 1e99;
      var.modifiable = true;
      var.recompile  = false;
      var.modified   = false;
      m_application->getUPSVars().push_back( var );

      initialized = true;
    }
#endif
  }
  
  //__________________________________
  //
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  //Construct the matl_set
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // for fileInfo variable
  d_zero_matl = scinew MaterialSubset();
  d_zero_matl->add(0);
  d_zero_matl->addReference();
}

//______________________________________________________________________
void MinMax::scheduleInitialize(SchedulerP& sched,
                                const LevelP& level)
{  
  printSchedule(level,cout_doing,"minMax::scheduleInitialize");
  Task* t = scinew Task("MinMax::initialize", 
                  this, &MinMax::initialize);
  
  t->computes(d_lb->lastCompTimeLabel );
  t->computes(d_lb->fileVarsStructLabel, d_zero_matl); 
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}
//______________________________________________________________________
void MinMax::initialize(const ProcessorGroup*, 
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing MinMax::initialize");
    
    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put(max_vartype(tminus), d_lb->lastCompTimeLabel );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;
    
    new_dw->put(fileInfo,    d_lb->fileVarsStructLabel, 0, patch);
    
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:MinMax  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

//______________________________________________________________________
void MinMax::scheduleRestartInitialize(SchedulerP   & sched,
                                       const LevelP & level)
{
  scheduleInitialize( sched, level);
}


//______________________________________________________________________
void MinMax::scheduleDoAnalysis(SchedulerP   & sched,
                                const LevelP & levelP)
{
  printSchedule(levelP,cout_doing,"MinMax::scheduleDoAnalysis");
   
  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_minMax", false, false, false, true, true); 
   
  Ghost::GhostType gn = Ghost::None;
  const Level* level = levelP.get_rep();
  const int L_indx = level->getIndex();
  
  //__________________________________
  //  computeMinMax task;     
  Task* t0 = scinew Task( "MinMax::computeMinMax", 
                          this,&MinMax::computeMinMax );

  sched_TimeVars( t0, levelP, d_lb->lastCompTimeLabel, false );
     
  for ( unsigned int i =0 ; i < d_analyzeVars.size(); i++ ) {
    VarLabel* label   = d_analyzeVars[i].label;
    const int myLevel = d_analyzeVars[i].level;
    
    // is this the right level for this variable?
    if ( isRightLevel( myLevel, L_indx, level ) ){
    
      // bulletproofing
      if( label == nullptr ){
        string name = label->getName();
        throw InternalError("MinMax: scheduleDoAnalysis label not found: " 
                           + name , __FILE__, __LINE__);
      }

      MaterialSubset* matSubSet = d_analyzeVars[i].matSubSet;
      t0->requires( Task::NewDW, label, matSubSet, gn, 0 );
     
      t0->computes( d_analyzeVars[i].reductionMinLabel, level, matSubSet );
      t0->computes( d_analyzeVars[i].reductionMaxLabel, level, matSubSet );
    }
  }
  
  sched->addTask( t0, level->eachPatch(), d_matl_set );
 
  //__________________________________
  //  Write min/max to a file
  // Only write data on patch 0 on each level

  Task* t1 = scinew Task( "MinMax::doAnalysis", 
                          this,&MinMax::doAnalysis );      
                            
  sched_TimeVars( t1, levelP, d_lb->lastCompTimeLabel, true );
  
  t1->requires( Task::OldDW, d_lb->fileVarsStructLabel, d_zero_matl, gn, 0 );
  
  // schedule the reduction variables
  for ( unsigned int i =0 ; i < d_analyzeVars.size(); i++ ) {
  
    int myLevel = d_analyzeVars[i].level;
    if ( isRightLevel( myLevel, L_indx, level) ){
    
      MaterialSubset* matSubSet = d_analyzeVars[i].matSubSet;
      
      t1->requires( Task::NewDW, d_analyzeVars[i].reductionMinLabel, level, matSubSet );
      t1->requires( Task::NewDW, d_analyzeVars[i].reductionMaxLabel, level, matSubSet );      
    }
  }

  t1->computes( d_lb->fileVarsStructLabel, d_zero_matl );
  
  // first patch on this level
  const Patch* p = level->getPatch(0);
  PatchSet* zeroPatch = scinew PatchSet();
  zeroPatch->add(p);
  zeroPatch->addReference();
  
  sched->addTask( t1, zeroPatch , d_matl_set );
  
  if( zeroPatch && zeroPatch->removeReference() ) {
    delete zeroPatch;
  }
  
}

//______________________________________________________________________
//  This task computes and min/max of each variable
//
void MinMax::computeMinMax(const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw)
{ 
  //__________________________________
  // compute min/max if it's time to write
  const Level* level = getLevel(patches);
  const int L_indx   = level->getIndex();
   
  if( isItTime( old_dw, level, d_lb->lastCompTimeLabel) == false ){
    return;
  }



  /*  Loop over patches  */
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);    

    /* Loop over variables */
    for (unsigned int i =0 ; i < d_analyzeVars.size(); i++) {
      VarLabel* label = d_analyzeVars[i].label;
      string labelName = label->getName();

      // bulletproofing
      if( label == nullptr ){
        throw InternalError("MinMax: analyze label not found: " 
                             + labelName , __FILE__, __LINE__);
      }    

      // Are we on the right level for this level
      const int myLevel = d_analyzeVars[i].level;
      if ( !isRightLevel( myLevel, L_indx, level) ){
        continue;
      }

      printTask(patches, patch,cout_doing,"Doing MinMax::computeMinMax");

      const TypeDescription* td = label->typeDescription();
      const TypeDescription* subtype = td->getSubType();

      int indx = d_analyzeVars[i].matl;

      switch(td->getType()){
        case TypeDescription::CCVariable:             // CC Variables
          switch(subtype->getType()) {

          case TypeDescription::double_type:{         // CC double
            GridIterator iter=patch->getCellIterator();
            findMinMax <constCCVariable<double>, double > ( new_dw, label, indx, patch, iter );
            break;
          }
          case TypeDescription::Vector: {             // CC Vector
            GridIterator iter=patch->getCellIterator();
            findMinMax< constCCVariable<Vector>, Vector > ( new_dw, label, indx, patch, iter );
            break;
          }
          default:
            throw InternalError("MinMax: invalid data type", __FILE__, __LINE__); 
          }
          break;

        case TypeDescription::NCVariable:             // NC Variables
          switch(subtype->getType()) {

          case TypeDescription::double_type:{         // NC double
            GridIterator iter=patch->getNodeIterator();
            findMinMax <constNCVariable<double>, double > ( new_dw, label, indx, patch, iter );
            break;
          }
          case TypeDescription::Vector: {             // NC Vector
            GridIterator iter=patch->getNodeIterator();
            findMinMax< constNCVariable<Vector>, Vector > ( new_dw, label, indx, patch, iter );
            break; 
          }
          default:
            throw InternalError("MinMax: invalid data type", __FILE__, __LINE__); 
          }
          break;            
        case TypeDescription::SFCXVariable: {         // SFCX double
          GridIterator iter=patch->getSFCXIterator();
          findMinMax <constSFCXVariable<double>, double > ( new_dw, label, indx, patch, iter );
          break;
        }
        case TypeDescription::SFCYVariable: {         // SFCY double
          GridIterator iter=patch->getSFCYIterator();
          findMinMax <constSFCYVariable<double>, double > ( new_dw, label, indx, patch, iter );
          break;
        }
        case TypeDescription::SFCZVariable: {         // SFCZ double
          GridIterator iter=patch->getSFCZIterator();
          findMinMax <constSFCZVariable<double>, double > ( new_dw, label, indx, patch, iter );
          break;
        }
        default:
          ostringstream warn;
          warn << "ERROR:AnalysisModule:MinMax: ("<< label->getName() << " " 
               << td->getName() << " ) has not been implemented" << endl;
          throw InternalError(warn.str(), __FILE__, __LINE__);
      }
    }  // VarLabel loop          
  }  // patches
}

//______________________________________________________________________
//  This task writes out the min/max of each VarLabel to a separate file.
void MinMax::doAnalysis(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  timeVars tv;
    
  getTimeVars( old_dw, level, d_lb->lastCompTimeLabel, tv );
  putTimeVars( new_dw, d_lb->lastCompTimeLabel, tv );
  
  if( tv.isItTime == false ){
    return;
  }
  
  //__________________________________
  //
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

    int proc =
      m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);
    
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write.  With AMR data the proc
    // may not own the patch
    if( proc == pg->myRank() ){  

      printTask(patches, patch,cout_doing,"Doing MinMax::doAnalysis");

      for (unsigned int i =0 ; i < d_analyzeVars.size(); i++) {
        VarLabel* label = d_analyzeVars[i].label;
        string labelName = label->getName();
         
        // bulletproofing
        if(label == nullptr){
          throw InternalError("MinMax: analyze label not found: " 
                          + labelName , __FILE__, __LINE__);
        }
        
        // Are we on the right level for this variable?
        const int myLevel = d_analyzeVars[i].level;
        
        if ( !isRightLevel( myLevel, L_indx, level ) ){
          continue;
        }

        //__________________________________
        // create the directory structure
        string minmaxDir = m_output->getOutputLocation() + "/MinMax";

        if( d_isDirCreated.count(minmaxDir) == 0){
          createDirectory(minmaxDir);
          d_isDirCreated.insert(minmaxDir);
        }
        
        ostringstream li;
        li<<"l"<<level->getIndex();
        string levelIndex = li.str();
        string levelDir = minmaxDir + "/" + levelIndex;

        if( d_isDirCreated.count(levelDir) == 0){
          createDirectory(levelDir);
          d_isDirCreated.insert(levelDir);
        }
        
        ostringstream fname;
        fname << levelDir << "/" << labelName << "_" << d_analyzeVars[i].matl;
        string filename = fname.str();

        //__________________________________
        //  Open the file pointer 
        //  if it's not in the fileInfo struct then create it
        FILE *fp;

        if( myFiles.count(filename) == 0 ){
          createFile(filename, fp, levelIndex);
          myFiles[filename] = fp;

        } else {
          fp = myFiles[filename];
        }
        if (!fp){
          throw InternalError("\nERROR:dataAnalysisModule:MinMax:  failed opening file"+filename,__FILE__, __LINE__);
        }   

        //__________________________________
        //  Now get the data from the DW and write it to the file
        string VLmax = labelName + "_max";
        string VLmin = labelName + "_min";
        const VarLabel* meMin = d_analyzeVars[i].reductionMinLabel;
        const VarLabel* meMax = d_analyzeVars[i].reductionMaxLabel;
        
        int indx = d_analyzeVars[i].matl;
        
        const TypeDescription* td = label->typeDescription();
        const TypeDescription* subtype = td->getSubType();

        switch(subtype->getType()) {

          case TypeDescription::double_type:{
            max_vartype maxQ;
            min_vartype minQ;
            
            new_dw->get( maxQ, meMax, level, indx);
            new_dw->get( minQ, meMin, level, indx);

            fprintf( fp, "%16.15E     %16.15E    %16.15E\n",tv.now, (double)minQ, (double)maxQ );
           break;
          }
          case TypeDescription::Vector: {
            maxvec_vartype maxQ;
            minvec_vartype minQ;
            
            new_dw->get( maxQ, meMax, level, indx);
            new_dw->get( minQ, meMin, level, indx);

            Vector maxQ_V = maxQ;
            Vector minQ_V = minQ;
            
            fprintf( fp, "%16.15E     [%16.15E %16.15E %16.15E]   [%16.15E %16.15E %16.15E]\n",tv.now,  
                          minQ_V.x(), minQ_V.y(), minQ_V.z(),maxQ_V.x(), maxQ_V.y(), maxQ_V.z() );
          
            break;
          }
        default:
          throw InternalError("MinMax: invalid data type", __FILE__, __LINE__); 
        }
        fflush(fp);
      }  // label names
    }  // time to write data
    
    // Put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents   
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo, d_lb->fileVarsStructLabel, 0, patch); 
  }  // patches
}


//______________________________________________________________________
//  Find the min/max of the VarLabel along with the 
//  position.  The position isn't used since we don't have a way 
//  to save that info in the DW.
template <class Tvar, class Ttype>
void MinMax::findMinMax( DataWarehouse*  new_dw,
                         const VarLabel* varLabel,
                         const int       indx,
                         const Patch*    patch,
                         GridIterator    iter )
{

  const Level* level = patch->getLevel();
  Tvar Q_var;
  Ttype Q;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);
  
  Ttype maxQ( Q_var[*iter] );  // initial values
  Ttype minQ( maxQ );
  
  IntVector maxIndx( *iter );
  IntVector minIndx( *iter );
  
  for (;!iter.done();iter++) {
    IntVector c = *iter;
    Q = Q_var[c];
    
    // use Max & Min instead of std::max & min
    // These functions can handle Vectors
    maxQ = Max(maxQ,Q);  
    minQ = Min(minQ,Q);
    
    if ( Q == maxQ ){
      maxIndx = c;
    }
    if (Q == minQ ){
      minIndx = c;
    }
  }  

  //Point maxPos = level->getCellPosition(maxIndx);
  //Point minPos = level->getCellPosition(minIndx);          

  // cout_doing << varLabel->getName() << " max: " << maxQ << " " << maxIndx << " maxPos " << maxPos << endl;
  // cout_doing << "         min: " << minQ << " " << minIndx << " minPos " << minPos << endl; 
  
  const string labelName = varLabel->getName();
  string VLmax = labelName + "_max";
  string VLmin = labelName + "_min";
  const VarLabel* meMin = VarLabel::find( VLmin );
  const VarLabel* meMax = VarLabel::find( VLmax );

  new_dw->put( ReductionVariable<Ttype, Reductions::Max<Ttype> >(maxQ), meMax, level, indx );
  new_dw->put( ReductionVariable<Ttype, Reductions::Min<Ttype> >(minQ), meMin, level, indx );
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void MinMax::createFile( string& filename,  
                         FILE*& fp, 
                         string& levelIndex)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }
  
  fp = fopen(filename.c_str(), "w");
  fprintf( fp,"#The reported min & max values are for this level %s \n", levelIndex.c_str() );
  fprintf( fp,"#Time                      min                       max\n" );
  
  cout_doing << d_myworld->myRank() << " MinMax:Created file " << filename << endl;

  cout << "OnTheFlyAnalysis MinMax results are located in " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   dirName/LevelIndex
void
MinMax::createDirectory(string& dirName)
{
  DIR *check = opendir(dirName.c_str());
  if ( check == nullptr ) {
    cout_doing << d_myworld->myRank() << " MinMax:Making directory " << dirName << endl;
    MKDIR( dirName.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
//______________________________________________________________________
//
bool MinMax::isRightLevel( const int myLevel, 
                           const int L_indx, 
                           const Level* level)
{
  if( myLevel == ALL_LEVELS || myLevel == L_indx )
    return true;
    
  int numLevels = level->getGrid()->numLevels();
  if( myLevel == FINEST_LEVEL && L_indx == numLevels -1 ){
    return true;
  }else{
    return false;
  }
}
