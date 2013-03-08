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

#include <CCA/Components/OnTheFlyAnalysis/particleExtract.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
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
//  setenv SCI_DEBUG "particleExtract_DBG_COUT:+" 
static DebugStream cout_doing("particleExtract_DOING_COUT", false);
static DebugStream cout_dbg("particleExtract_DBG_COUT", false);
//______________________________________________________________________              
particleExtract::particleExtract(ProblemSpecP& module_spec,
                         SimulationStateP& sharedState,
                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  ps_lb = scinew particleExtractLabel();
  M_lb = scinew MPMLabel();
}

//__________________________________
particleExtract::~particleExtract()
{
  cout_doing << " Doing: destorying particleExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  VarLabel::destroy(ps_lb->filePointerLabel);
  VarLabel::destroy(ps_lb->filePointerLabel_preReloc);
  delete ps_lb;
  delete M_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void particleExtract::problemSetup(const ProblemSpecP& prob_spec,
                                   const ProblemSpecP& ,
                                   GridP& grid,
                                   SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tparticleExtract" << endl;

  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  
  if(!d_dataArchiver){
    throw InternalError("particleExtract:couldn't get output port", __FILE__, __LINE__);
  }
  
  vector<int> m;
  m.push_back( d_matl->getDWIndex() );
  
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());
  
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();   
  
  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());
                                            
   ps_lb->filePointerLabel  =  VarLabel::create("filePointer", 
                                            ParticleVariable< FILE* >::getTypeDescription() );
   ps_lb->filePointerLabel_preReloc  =  VarLabel::create("filePointer+", 
                                            ParticleVariable< FILE* >::getTypeDescription() );
                                             
  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_StartTime);            
  d_prob_spec->require("timeStop",          d_StopTime);

  d_prob_spec->require("colorThreshold",    d_colorThreshold);
  //__________________________________
  //  Read in variables label names
  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("particleExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);    
  } 
  map<string,string> attribute;                    
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    var_spec->getAttributes(attribute);
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("particleExtract: analyze label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();
    
    //__________________________________
    // Bulletproofing
    bool throwException = false;  
    
    // only certain particle types can be extracted
    if( td->getType() != TypeDescription::ParticleVariable ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::int_type    &&
          subtype->getType() != TypeDescription::Vector      &&
          subtype->getType() != TypeDescription::Matrix3 ) ) {
      throwException = true;
    }
    if( throwException ){       
      ostringstream warn;
      warn << "ERROR:AnalysisModule:particleExtact: ("<<label->getName() << " " 
           << td->getName() << " ) is either not a particle variable "
           << "or a valid type (int double, Vector)" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    d_varLabels.push_back(label);
  }    
 
  // Start time < stop time
  if(d_StartTime > d_StopTime){
    throw ProblemSetupException("\n ERROR:particleExtract: startTime > stopTime. \n", __FILE__, __LINE__);
  }
 
 // Tell the shared state that these variable need to be relocated
  int matl = d_matl->getDWIndex();
  sharedState->d_particleState_preReloc[matl].push_back(ps_lb->filePointerLabel_preReloc);
  sharedState->d_particleState[matl].push_back(ps_lb->filePointerLabel);
  
  //__________________________________
  //  Warning
  proc0cout << "\n\n______________________________________________________________________" << endl;
  proc0cout << "  WARNING      WARNING       WARNING" << endl;
  proc0cout << "     DataAnalysis:particleExract" << endl;
  proc0cout << "         BE VERY JUDICIOUS when selecting the <samplingFrequency> " << endl;
  proc0cout << "         and the number of particles to extract data from. Every time" << endl;
  proc0cout << "         the particles are analyized N particle files are opened and closed" << endl;
  proc0cout << "         This WILL slow your simulation down!" << endl;
  proc0cout << "______________________________________________________________________\n\n" << endl;  
  
  
}

//______________________________________________________________________
void particleExtract::scheduleInitialize(SchedulerP& sched,
                                         const LevelP& level)
{
  cout_doing << "particleExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("particleExtract::initialize", 
                  this, &particleExtract::initialize);
  
  t->computes( ps_lb->lastWriteTimeLabel );
  t->computes( ps_lb->filePointerLabel ) ;
  sched->addTask( t, level->eachPatch(), d_matl_set );
}
//______________________________________________________________________
void particleExtract::initialize(const ProcessorGroup*, 
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tparticleExtract" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
     
    double tminus = -1.0/d_writeFreq;
    new_dw->put( max_vartype( tminus ), ps_lb->lastWriteTimeLabel );
    
    ParticleVariable<FILE*> myFiles;
    int indx = d_matl->getDWIndex(); 
    ParticleSubset* pset = new_dw->getParticleSubset( indx, patch );
    new_dw->allocateAndPut( myFiles, ps_lb->filePointerLabel, pset );
    
    for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      myFiles[idx] = NULL;
    }
    
    
    //__________________________________
    //bullet proofing
    if( ! new_dw->exists(M_lb->pColorLabel, indx, patch ) ){
      ostringstream warn;
      warn << "ERROR:particleExtract  In order to use the DataAnalysis Module particleExtract "
           << "you must 'color' least one MPM geom_object.";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:particleExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    } 
  }  
}

void particleExtract::restartInitialize()
{
// need to do something here
//  new_dw->put(max_vartype(0.0), ps_lb->lastWriteTimeLabel);
}
//______________________________________________________________________
void particleExtract::scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                         const LevelP& level)
{ 
  int L_indx = level->getIndex();
  if(!doMPMOnLevel(L_indx,level->getGrid()->numLevels())){
    return;
  }

  cout_doing<< "particleExtract::scheduleDoAnalysis_preReloc " << endl;
  Task* t = scinew Task("particleExtract::doAnalysis_preReloc", 
                   this,&particleExtract::doAnalysis_preReloc);

  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.  Put it here so it will be registered during a restart
  sched->overrideVariableBehavior("filePointer", false, false, false, true, true);
                     
  Ghost::GhostType gn = Ghost::None;
  t->requires( Task::OldDW,  ps_lb->filePointerLabel, gn, 0 );
  t->computes( ps_lb->filePointerLabel_preReloc  );
  
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}
//______________________________________________________________________
void particleExtract::doAnalysis_preReloc(const ProcessorGroup* pg,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();
    
    ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
    constParticleVariable<FILE*>myFiles;
    ParticleVariable<FILE*> myFiles_preReloc;

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
        myFiles_preReloc[idx] = NULL;
      }
    }
  }
}   
 
//______________________________________________________________________
void particleExtract::scheduleDoAnalysis(SchedulerP& sched,
                                         const LevelP& level)
{
  // only schedule task on the finest level
  int L_indx = level->getIndex();
  if(!doMPMOnLevel(L_indx,level->getGrid()->numLevels())){
    return;
  }

  cout_doing << "particleExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("particleExtract::doAnalysis", 
                   this,&particleExtract::doAnalysis);
                     
                     
  t->requires(Task::OldDW, ps_lb->lastWriteTimeLabel);
  
  Ghost::GhostType gn = Ghost::None;
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == NULL){
      string name = d_varLabels[i]->getName();
      throw InternalError("particleExtract: scheduleDoAnalysis label not found: " 
                          + name , __FILE__, __LINE__);
    }
    t->requires(Task::NewDW,d_varLabels[i], gn, 0);
  }
  t->requires( Task::NewDW,  M_lb->pXLabel,           gn );
  t->requires( Task::NewDW,  M_lb->pParticleIDLabel,  gn );
  t->requires( Task::NewDW,  M_lb->pColorLabel,       gn );
  t->requires( Task::NewDW,  ps_lb->filePointerLabel, gn );
  
  t->computes( ps_lb->lastWriteTimeLabel );
  t->modifies( ps_lb->filePointerLabel );
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void particleExtract::doAnalysis(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{   
  UintahParallelComponent* DA = dynamic_cast<UintahParallelComponent*>(d_dataArchiver);
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>( DA->getPort("load balancer"));
    
  const Level* level = getLevel(patches);
  
  max_vartype writeTime;
  old_dw->get(writeTime, ps_lb->lastWriteTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    int proc = lb->getPatchwiseProcessorAssignment(patch);
    cout_dbg << Parallel::getMPIRank() << "   working on patch " << patch->getID() << " which is on proc " << proc << endl;
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myrank() && now >= nextWriteTime){
    
     cout_doing << pg->myrank() << " " 
                << "Doing doAnalysis (particleExtract)\t\t\t\tL-"
                << level->getIndex()
                << " patch " << patch->getGridIndex()<< endl;
      //__________________________________
      // loop over each of the variables
      // load them into the data vectors
      vector< constParticleVariable<int> >      integer_data;
      vector< constParticleVariable<double> >   double_data;
      vector< constParticleVariable<Vector> >   Vector_data;
      vector< constParticleVariable<Matrix3> >  Matrix3_data;
      
      constParticleVariable<int>    p_integer;      
      constParticleVariable<double> p_double;
      constParticleVariable<Vector> p_Vector;
      constParticleVariable<Matrix3> p_Matrix3; 
      constParticleVariable<long64> pid;
      constParticleVariable<Point> px;  
      constParticleVariable<double>pColor;

            
      Ghost::GhostType  gn = Ghost::None;
      int NGP = 0;
      int indx = d_matl->getDWIndex(); 
      ParticleSubset* pset = new_dw->getParticleSubset(indx, patch,
                                                 gn, NGP, M_lb->pXLabel);
     
      // additional particle data
      new_dw->get(pid,    M_lb->pParticleIDLabel, pset);
      new_dw->get(px,     M_lb->pXLabel,          pset);
      new_dw->get(pColor, M_lb->pColorLabel,      pset);
      
      // file pointers
      ParticleVariable<FILE*>myFiles;
      new_dw->getModifiable( myFiles,    ps_lb->filePointerLabel, pset );
      
      //__________________________________
      //  Put particle data into arrays <double,int,....>_data
      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
        
        // bulletproofing
        if(d_varLabels[i] == NULL){
          string name = d_varLabels[i]->getName();
          throw InternalError("particleExtract: analyze label not found: " 
                          + name , __FILE__, __LINE__);
        }

        const TypeDescription* td = d_varLabels[i]->typeDescription();
        const TypeDescription* subtype = td->getSubType();
        
        switch(td->getType()){
          case TypeDescription::ParticleVariable:    
            switch(subtype->getType()) {
            
            case TypeDescription::double_type:
              new_dw->get(p_double, d_varLabels[i], pset);
              double_data.push_back(p_double);
              break;
             
            case TypeDescription::Vector:
              new_dw->get(p_Vector, d_varLabels[i], pset);
              Vector_data.push_back(p_Vector);
              break;
              
            case TypeDescription::int_type:
              new_dw->get(p_integer, d_varLabels[i], pset);
              integer_data.push_back(p_integer);
              break; 
              
            case TypeDescription::Matrix3:
              new_dw->get(p_Matrix3, d_varLabels[i], pset);
              Matrix3_data.push_back(p_Matrix3);
              break;
            default:
              throw InternalError("particleExtract: invalid data type", __FILE__, __LINE__); 
            }
            break;
        default:
          ostringstream warn;
          warn << "ERROR:AnalysisModule:lineExtact: ("<<d_varLabels[i]->getName() << " " 
               << td->getName() << " ) has not been implemented" << endl;
          throw InternalError(warn.str(), __FILE__, __LINE__);
        }
      }            
      
      // create the directory structure
      string udaDir = d_dataArchiver->getOutputLocation();
      string pPath = udaDir + "/particleExtract";

      ostringstream li;
      li<<"L-"<<level->getIndex();
      string levelIndex = li.str();
      string path = pPath + "/" + levelIndex;
      
      if( d_isDirCreated.count(path) == 0){
        createDirectory(pPath, levelIndex);
        d_isDirCreated.insert(path);
      }
        

      //__________________________________
      // loop over the particle
      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if (pColor[idx] > d_colorThreshold){
        
          ostringstream fname;
          fname<<path<<"/"<<pid[idx];
          string filename = fname.str();
          
          // open the file
          FILE *fp = NULL;
          createFile(filename,fp);
          
          //__________________________________
          //   HACK: don't keep track of the file pointers.
          //   create the file every pass through.  See message below.            
#if 0          
          if( myFiles[idx] ){           // if the filepointer has been previously stored.
            fp = myFiles[idx];
            cout << Parallel::getMPIRank() << " I think this pointer is valid " << idx << " fp " << fp << " patch " << patch->getID() << endl;
          } else {
            createFile(filename, fp);
            myFiles[idx] = fp;
          }
#endif         
          
          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:particleExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }

          // write particle position and time
          double time = d_dataArchiver->getCurrentTime();
          fprintf(fp,    "%E\t %E\t %E\t %E",time, px[idx].x(),px[idx].y(),px[idx].z());


           // WARNING  If you change the order that these are written out you must 
           // also change the order that the header is written

          // write <int> variables      
          for (unsigned int i=0 ; i <  integer_data.size(); i++) {
            fprintf(fp, "    %i",integer_data[i][idx]);            
          }          
          // write <double> variables
          for (unsigned int i=0 ; i <  double_data.size(); i++) {
            fprintf(fp, "    %16E",double_data[i][idx]);            
          }
          // write <Vector> variable
          for (unsigned int i=0 ; i <  Vector_data.size(); i++) {
            fprintf(fp, "    % 16E      %16E      %16E",
                    Vector_data[i][idx].x(),
                    Vector_data[i][idx].y(),
                    Vector_data[i][idx].z() );            
          } 
          // write <Matrix3> variable
          for (unsigned int i=0 ; i <  Matrix3_data.size(); i++) {
            for (int row = 0; row<3; row++){
              fprintf(fp, "    % 16E      %16E      %16E",
                      Matrix3_data[i][idx](row,0),
                      Matrix3_data[i][idx](row,1),
                      Matrix3_data[i][idx](row,2) );
            }            
          }        

          fprintf(fp,    "\n");
          
          //__________________________________
          //  HACK:  Close each file and set the fp to NULL
          //  Remove this hack once we figure out how to use
          //  particle relocation to move file pointers between
          //  patches.
          fclose(fp);
          myFiles[idx] == NULL;
        }
      }  // loop over particles
      lastWriteTime = now;     
    }  // time to write data
    
   new_dw->put(max_vartype(lastWriteTime), ps_lb->lastWriteTimeLabel); 
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void particleExtract::createFile(string& filename, FILE*& fp)
{ 
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }
  
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"# Time    X      Y      Z     "); 
  
  // All ParticleVariable<int>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const TypeDescription* td = d_varLabels[i]->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    if(subtype->getType() == TypeDescription::int_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s", name.c_str());
    }
  }
  // All ParticleVariable<double>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const TypeDescription* td = d_varLabels[i]->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    if(subtype->getType() == TypeDescription::double_type){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s", name.c_str());
    }
  }
  // All ParticleVariable<Vector>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const TypeDescription* td = d_varLabels[i]->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    if(subtype->getType() == TypeDescription::Vector){
      string name = d_varLabels[i]->getName(); 
      fprintf(fp,"     %s.x      %s.y      %s.z", name.c_str(),name.c_str(),name.c_str());
    }
  }
  // All ParticleVariable<Matrix3>
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    const TypeDescription* td = d_varLabels[i]->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    if(subtype->getType() == TypeDescription::Matrix3){
      string name = d_varLabels[i]->getName(); 
      for (int row = 0; row<3; row++){
        fprintf(fp,"     %s(%i,0)      %s(%i,1)      %s(%i,2)", name.c_str(),row,name.c_str(),row,name.c_str(),row);
      }
    }
  }
  fprintf(fp,"\n");
  fflush(fp);

  cout << Parallel::getMPIRank() << " particleExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   dirName/LevelIndex
//
void
particleExtract::createDirectory(string& dirName, string& levelIndex)
{
  DIR *check = opendir(dirName.c_str());
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << "particleExtract:Making directory " << dirName << endl;
    MKDIR( dirName.c_str(), 0777 );
  } else {
    closedir(check);
  }
  
  // level index
  string path = dirName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL ) {
    cout << "particleExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
//______________________________________________________________________
//
bool
particleExtract::doMPMOnLevel(int level, int numLevels)
{
  int minGridLevel = 0;
  int maxGridLevel = 1000;
  return (level >= minGridLevel && level <= maxGridLevel) ||
          (minGridLevel < 0 && level == numLevels + minGridLevel);
}
