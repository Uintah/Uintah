/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/planeAverage.h>
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
//  setenv SCI_DEBUG "planeAverage:+"
static DebugStream do_cout("planeAverage", "OnTheFlyAnalysis", "planeAverage debug stream", false);

//______________________________________________________________________
/*
     This module computes the spatial average of a variable over a plane
TO DO:


______________________________________________________________________*/

planeAverage::planeAverage( const ProcessorGroup    * myworld,
                            const MaterialManagerP    materialManager,
                            const ProblemSpecP      & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set  = nullptr;
  d_zero_matl = nullptr;
  d_lb        = scinew planeAverageLabel();
}

//__________________________________
planeAverage::~planeAverage()
{
  do_cout << " Doing: destorying planeAverage " << endl;
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
void planeAverage::problemSetup(const ProblemSpecP&,
                                const ProblemSpecP&,
                                GridP& grid,
                                std::vector<std::vector<const VarLabel* > > &PState,
                                std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  do_cout << "Doing problemSetup \t\t\t\tplaneAverage" << endl;

  int numMatls  = m_materialManager->getNumMatls();

  d_lb->lastCompTimeLabel =  VarLabel::create("lastCompTime_planeAvg",
                                              max_vartype::getTypeDescription() );

  d_lb->fileVarsStructLabel = VarLabel::create("FileInfo_planeAvg",
                                               PerPatch<FileInfoP>::getTypeDescription() );

  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", d_writeFreq);
  m_module_spec->require("timeStart",         d_startTime);
  m_module_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("planeAverage: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  //__________________________________
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
  //  Plane orientation
  string orient;
  m_module_spec->require("planeOrientation", orient);
  if ( orient == "XY" ){
    d_planeOrientation = XY;
  } else if ( orient == "XZ" ) {
    d_planeOrientation = XZ;
  } else if ( orient == "YZ" ) {
    d_planeOrientation = YZ;
  }

  //__________________________________
  //  Now loop over all the variables to be analyzed

  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {

    var_spec->getAttributes( attribute );

    //__________________________________
    // Read in the variable name
    string labelName = attribute["label"];
    VarLabel* label = VarLabel::find(labelName);
    if( label == nullptr ){
      throw ProblemSetupException("planeAverage: analyze label not found: " + labelName , __FILE__, __LINE__);
    }

    // Bulletproofing - The user must specify the matl for single matl
    // variables
    if ( labelName == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("planeAverage: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
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
      throw ProblemSetupException("planeAverage: Invalid material index specified for a variable", __FILE__, __LINE__);
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
      warn << "ERROR:AnalysisModule:planeAverage: ("<<label->getName() << " "
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    //  Create the max VarLabels for the reduction variables
   
    // double
    if( subType == TypeDescription::double_type ) {
    }
    // Vectors
    if( subType == TypeDescription::Vector ) {
    }

    varProperties me;
    me.label = label;
    me.matl  = matl;
    me.level = level;
    d_analyzeVars.push_back(me);
    
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
void planeAverage::scheduleInitialize(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,do_cout,"planeAverage::scheduleInitialize");

  Task* t = scinew Task("planeAverage::initialize",
                  this, &planeAverage::initialize);

  t->computes(d_lb->lastCompTimeLabel );
  t->computes(d_lb->fileVarsStructLabel, d_zero_matl);
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}

//______________________________________________________________________
void planeAverage::initialize(const ProcessorGroup  *,
                              const PatchSubset     * patches,
                              const MaterialSubset  *,
                              DataWarehouse         *,
                              DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,do_cout,"Doing planeAverage::initialize");

    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), d_lb->lastCompTimeLabel );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo,    d_lb->fileVarsStructLabel, 0, patch);

    if( patch->getGridIndex() == 0 ){   // only need to do this once
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:planeAverage  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}
//______________________________________________________________________
//
void planeAverage::restartInitialize()
{
}

//______________________________________________________________________
void planeAverage::scheduleDoAnalysis(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,do_cout,"planeAverage::scheduleDoAnalysis");

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_planeAvg", false, false, false, true, true);

  Ghost::GhostType gn = Ghost::None;
  const int L_indx = level->getIndex();

  //__________________________________
  //  compute the planar average task;
  Task* t0 = scinew Task( "planeAverage::computeAverage",
                     this,&planeAverage::computeAverage );

  t0->requires( Task::OldDW, m_simulationTimeLabel );
  t0->requires( Task::OldDW, d_lb->lastCompTimeLabel );

  for ( unsigned int i =0 ; i < d_analyzeVars.size(); i++ ) {
    VarLabel* label   = d_analyzeVars[i].label;
    const int myLevel = d_analyzeVars[i].level;

    // is this the right level for this variable?
    if ( isRightLevel( myLevel, L_indx, level ) ){

      // bulletproofing
      if( label == nullptr ){
        string name = label->getName();
        throw InternalError("planeAverage: scheduleDoAnalysis label not found: "
                           + name , __FILE__, __LINE__);
      }

      MaterialSubset* matSubSet = scinew MaterialSubset();
      matSubSet->add( d_analyzeVars[i].matl );
      matSubSet->addReference();

      t0->requires( Task::NewDW, label, matSubSet, gn, 0 );

      if(matSubSet && matSubSet->removeReference()){
        delete matSubSet;
      }
    }
  }

  sched->addTask( t0, level->eachPatch(), d_matl_set );

  //__________________________________
  //  Write max to a file
  // Only write data on patch 0 on each level

  Task* t1 = scinew Task( "planeAverage::doAnalysis",
                     this,&planeAverage::doAnalysis );

  t1->requires( Task::OldDW, m_simulationTimeLabel );
  t1->requires( Task::OldDW, d_lb->lastCompTimeLabel );
  t1->requires( Task::OldDW, d_lb->fileVarsStructLabel, d_zero_matl, gn, 0 );

  // schedule the reduction variables
  for ( unsigned int i =0 ; i < d_analyzeVars.size(); i++ ) {

    int myLevel = d_analyzeVars[i].level;
    if ( isRightLevel( myLevel, L_indx, level) ){
      MaterialSubset* matSubSet = scinew MaterialSubset();
      matSubSet->add( d_analyzeVars[i].matl );
      matSubSet->addReference();
    }
  }

  t1->computes( d_lb->lastCompTimeLabel );
  t1->computes( d_lb->fileVarsStructLabel, d_zero_matl );

  // first patch on this level
  const Patch* p = level->getPatch(0);
  PatchSet* zeroPatch = scinew PatchSet();
  zeroPatch->add(p);
  zeroPatch->addReference();

  sched->addTask( t1, zeroPatch , d_matl_set );
}

//______________________________________________________________________
//  This task computes and min/max of each variable
//
void planeAverage::computeAverage(const ProcessorGroup * pg,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        * old_dw,
                                  DataWarehouse        * new_dw)
{

  // the user may want to restart from an uda that wasn't using the DA module
  // This logic allows that.
  max_vartype writeTime;
  double lastWriteTime = 0;
  if( old_dw->exists( d_lb->lastCompTimeLabel ) ){
    old_dw->get(writeTime, d_lb->lastCompTimeLabel);
    lastWriteTime = writeTime;
  }

  // double now = m_materialManager->getElapsedSimTime();

  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  // Get the delta T from the warehouse so time includes the current
  // time step.
  DataWarehouse * dw = m_scheduler->get_dw(0);

  delt_vartype delt_var;

cout << " now: " << now << " delT: " << delt_var << endl;

  if( dw->exists( m_delTLabel ) ) {
    m_scheduler->get_dw(0)->get( delt_var, m_delTLabel );
    now += delt_var;
  }

  if(now < d_startTime || now > d_stopTime){
    return;
  }

  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  //__________________________________
  // compute plane average if it's time to write
  if( now >= nextWriteTime){

    const LevelP level = getLevelP(patches);
    const int L_indx = level->getIndex();

    /*  Loop over patches  */
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      /* Loop over variables */
      for (unsigned int i =0 ; i < d_analyzeVars.size(); i++) {
        VarLabel* label = d_analyzeVars[i].label;
        string labelName = label->getName();

        // bulletproofing
        if( label == nullptr ){
          throw InternalError("planeAverage: analyze label not found: "
                               + labelName , __FILE__, __LINE__);
        }

        // Are we on the right level for this level
        const int myLevel = d_analyzeVars[i].level;
        if ( !isRightLevel( myLevel, L_indx, level) ){
          continue;
        }

        printTask(patches, patch,do_cout,"Doing planeAverage::computeAverage");

        const TypeDescription* td = label->typeDescription();
        const TypeDescription* subtype = td->getSubType();

        int indx = d_analyzeVars[i].matl;

        switch(td->getType()){
          case TypeDescription::CCVariable:             // CC Variables
            switch(subtype->getType()) {

            case TypeDescription::double_type:{         // CC double
              GridIterator iter=patch->getCellIterator();
              findAverage <constCCVariable<double>, double > ( new_dw, label, indx, patch, iter );
              break;
            }
            case TypeDescription::Vector: {             // CC Vector
              GridIterator iter=patch->getCellIterator();
              findAverage< constCCVariable<Vector>, Vector > ( new_dw, label, indx, patch, iter );
              break;
            }
            default:
              throw InternalError("planeAverage: invalid data type", __FILE__, __LINE__);
            }
            break;

          case TypeDescription::NCVariable:             // NC Variables
            switch(subtype->getType()) {

            case TypeDescription::double_type:{         // NC double
              GridIterator iter=patch->getNodeIterator();
              findAverage <constNCVariable<double>, double > ( new_dw, label, indx, patch, iter );
              break;
            }
            case TypeDescription::Vector: {             // NC Vector
              GridIterator iter=patch->getNodeIterator();
              findAverage< constNCVariable<Vector>, Vector > ( new_dw, label, indx, patch, iter );
              break;
            }
            default:
              throw InternalError("planeAverage: invalid data type", __FILE__, __LINE__);
            }
            break;
          case TypeDescription::SFCXVariable: {         // SFCX double
            GridIterator iter=patch->getSFCXIterator();
            findAverage <constSFCXVariable<double>, double > ( new_dw, label, indx, patch, iter );
            break;
          }
          case TypeDescription::SFCYVariable: {         // SFCY double
            GridIterator iter=patch->getSFCYIterator();
            findAverage <constSFCYVariable<double>, double > ( new_dw, label, indx, patch, iter );
            break;
          }
          case TypeDescription::SFCZVariable: {         // SFCZ double
            GridIterator iter=patch->getSFCZIterator();
            findAverage <constSFCZVariable<double>, double > ( new_dw, label, indx, patch, iter );
            break;
          }
          default:
            ostringstream warn;
            warn << "ERROR:AnalysisModule:planeAverage: ("<< label->getName() << " "
                 << td->getName() << " ) has not been implemented" << endl;
            throw InternalError(warn.str(), __FILE__, __LINE__);
        }
      }  // VarLabel loop
    }  // patches
  }  // time to write data
}

//______________________________________________________________________
//  This task writes out the plane average of each VarLabel to a separate file.
void planeAverage::doAnalysis(const ProcessorGroup* pg,
                              const PatchSubset   * patches,
                              const MaterialSubset*,
                              DataWarehouse       * old_dw,
                              DataWarehouse       * new_dw)
{
  const Level* level = getLevel(patches);
  const LevelP levelP = getLevelP(patches);
  int L_indx = level->getIndex();

  max_vartype writeTime;
  double lastWriteTime = 0;
  if( old_dw->exists( d_lb->lastCompTimeLabel ) ){
    old_dw->get(writeTime, d_lb->lastCompTimeLabel);
    lastWriteTime = writeTime;
  }

  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  // Get the delta t from the warehouse so time includes the current time step.
  DataWarehouse * dw = m_scheduler->get_dw(0);

  delt_vartype delt_var;
  if( dw->exists( m_delTLabel ) ) {
    m_scheduler->get_dw(0)->get( delt_var, m_delTLabel );
    now += delt_var;
  }

  if(now < d_startTime || now > d_stopTime){
    new_dw->put(max_vartype(lastWriteTime), d_lb->lastCompTimeLabel);
    return;
  }

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

    int proc =
      m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);

    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write.  With AMR data the proc
    // may not own the patch
    if( proc == pg->myRank() && now >= nextWriteTime){

      printTask(patches, patch,do_cout,"Doing planeAverage::doAnalysis");

      for (unsigned int i =0 ; i < d_analyzeVars.size(); i++) {
        VarLabel* label = d_analyzeVars[i].label;
        string labelName = label->getName();

        // bulletproofing
        if(label == nullptr){
          throw InternalError("planeAverage: analyze label not found: "
                          + labelName , __FILE__, __LINE__);
        }

        // Are we on the right level for this variable?
        const int myLevel = d_analyzeVars[i].level;

        if ( !isRightLevel( myLevel, L_indx, levelP ) ){
          continue;
        }

        //__________________________________
        //
        // create the directory structure including sub directories
        string udaDir = m_output->getOutputLocation();
   
        timeStep_vartype timeStep_var;      
        old_dw->get( timeStep_var, m_timeStepLabel );
        int ts = timeStep_var;

        ostringstream tname;
        tname << "t" << std::setw(5) << std::setfill('0') << ts;
        string timestep = tname.str();
        
        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();
        
        string path = "planeAverage/" + timestep + "/" + levelIndex;
        
        if( d_isDirCreated.count( path ) == 0 ){
          createDirectory( 0777, udaDir,  path );
          d_isDirCreated.insert( path );
        } 
        
        ostringstream fname;
        fname << udaDir << "/" << path << "/" << labelName << "_" << d_analyzeVars[i].matl;
        string filename = fname.str();
        
        //__________________________________
        //  Open the file pointer
        //  if it's not in the fileInfo struct then create it
        FILE *fp;
        if( myFiles.count(filename) == 0 ){
          createFile( filename, fp, levelIndex );
          myFiles[filename] = fp;

        } else {
          fp = myFiles[filename];
        }
        if (!fp){
          throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed opening file: "+filename,__FILE__, __LINE__);
        }

        //__________________________________
        //  Now get the data from the DW and write it to the file

        int indx = d_analyzeVars[i].matl;

        const TypeDescription* td = label->typeDescription();
        const TypeDescription* subtype = td->getSubType();

        switch(subtype->getType()) {

          case TypeDescription::double_type:{
            double aveQ;

            fprintf( fp, "%16.15E     %16.15E \n",now, (double)aveQ );
           break;
          }
          case TypeDescription::Vector: {
            Vector aveQ_V;

            fprintf( fp, "%16.15E     [%16.15E %16.15E %16.15E] \n",now,
                          aveQ_V.x(), aveQ_V.y(), aveQ_V.z() );

            break;
          }
        default:
          throw InternalError("planeAverage: invalid data type", __FILE__, __LINE__);
        }
        fflush(fp);
      }  // label names

      lastWriteTime = now;
    }  // time to write data

    // Put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo,                   d_lb->fileVarsStructLabel, 0, patch);
    new_dw->put(max_vartype(lastWriteTime), d_lb->lastCompTimeLabel );
  }  // patches
}

//______________________________________________________________________
//  Find the average of the VarLabel
template <class Tvar, class Ttype>
void planeAverage::findAverage( DataWarehouse * new_dw,
                                const VarLabel* varLabel,
                                const int       indx,
                                const Patch   * patch,
                                GridIterator    iter )
{
  cout <<"   " << varLabel->getName() << endl;

  Tvar Q_var;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);
  
  IntVector lo;
  IntVector hi;
  planeIterator( iter, lo, hi );

  cout <<"   iter.begin: " << iter.begin() << " lo: " << lo << endl;
  cout <<"   iter.end:   " << iter.end() <<   " hi: " << hi << endl;
  
  for ( auto z = lo.z(); z<hi.z(); z++ ) {          // This is the loop over planes
  
    Ttype sum( 0 );  // initial values
      
    for ( auto y = lo.y(); y<hi.y(); y++ ) {        // cells in the plane
      for ( auto x = lo.x(); x<hi.x(); x++ ) {
        IntVector c(x,y,z);
        
        switch( d_planeOrientation ){
          case XY:{
            c = IntVector(x,y,z);
            break;
          }
          case XZ:{
            c = IntVector(x,z,y);
            break;
          }
          case YZ:{
            c = IntVector(y,z,x);
            break;
          }
          default:
            break;
        }
//        cout << "   c: " << c << endl;
        sum = sum + Q_var[c];
      }
    }

    const string labelName = varLabel->getName();
    do_cout << labelName << "plane: " << z << " sum: " <<sum << endl;
    
    
//    string VLmax = labelName + "_max";
//    const VarLabel* meMax = VarLabel::find( VLmax );

//    new_dw->put( ReductionVariable<Ttype, Reductions::Max<Ttype> >(maxQ), meMax, level, indx );

  }
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void planeAverage::createFile(string  & filename,  
                              FILE*   & fp, 
                              string  & levelIndex)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }

  fp = fopen(filename.c_str(), "w");
  
  if (!fp){
    perror("Error opening file:");
    throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed opening file: " + filename,__FILE__, __LINE__);
  }
  
  fprintf( fp,"# Level-%s \n", levelIndex.c_str() );
  fprintf( fp,"# Time            Plane location            Average\n" );
  cout << "OnTheFlyAnalysis planeAverage results are located in " << filename << endl;
}

//______________________________________________________________________
// create a series of sub directories below the rootpath.
int
planeAverage::createDirectory( mode_t mode, 
                               const std::string & rootPath,  
                               std::string       & subDirs )
{
  struct stat st;

  do_cout << d_myworld->myRank() << " planeAverage:Making directory " << subDirs << endl;
  
  for( std::string::iterator iter = subDirs.begin(); iter != subDirs.end(); ){

    string::iterator newIter = std::find( iter, subDirs.end(), '/' );
    std::string newPath = rootPath + "/" + std::string( subDirs.begin(), newIter);

    // does path exist
    if( stat( newPath.c_str(), &st) != 0 ){ 
    
      int rc = mkdir( newPath.c_str(), mode);
      
      // bulletproofing     
      if(  rc != 0 && errno != EEXIST ){
        cout << "cannot create folder [" << newPath << "] : " << strerror(errno) << endl;
        throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed creating dir: "+newPath,__FILE__, __LINE__);
      }
    }
    else {      
      if( !S_ISDIR( st.st_mode ) ){
        errno = ENOTDIR;
        cout << "path [" << newPath << "] not a dir " << endl;
        return -1;
      } else {
        cout << "path [" << newPath << "] already exists " << endl;
      }
    }

    iter = newIter;
    if( newIter != subDirs.end() ){
      ++ iter;
    }
  }
  return 0;
}







//______________________________________________________________________
//
bool planeAverage::isRightLevel(const int myLevel, const int L_indx, const LevelP& level)
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
//______________________________________________________________________
//
void planeAverage::planeIterator( const GridIterator& patchIter,
                                  IntVector & lo,
                                  IntVector & hi )
{
  IntVector patchLo = patchIter.begin();
  IntVector patchHi = patchIter.end();

  switch( d_planeOrientation ){
    case XY:{
      lo = patchLo;
      hi = patchHi;
      break;
    }
    case XZ:{
      lo.x( patchLo.x() );
      lo.y( patchLo.z() );
      lo.z( patchLo.y() );

      hi.x( patchHi.x() );
      hi.y( patchHi.z() );
      hi.z( patchHi.y() );
      break;
    }
    case YZ:{
      lo.x( patchLo.y() );
      lo.y( patchLo.z() );
      lo.z( patchLo.x() );

      hi.x( patchHi.y() );
      hi.y( patchHi.z() );
      hi.z( patchHi.x() );
      break;
    }
    default:
      break;
  }
}
