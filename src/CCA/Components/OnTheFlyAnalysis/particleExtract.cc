/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/DbgOutput.h>

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
//  setenv SCI_DEBUG "particleExtract:+"

Dout dout_OTF_partEx("particleExtract",     "OnTheFlyAnalysis", "Task scheduling and execution.", false);
Dout dbg_OTF_partEx( "particleExtract_dbg", "OnTheFlyAnalysis", "Display detailed debugging info.", false);

//______________________________________________________________________
particleExtract::particleExtract(const ProcessorGroup* myworld,
                                 const MaterialManagerP materialManager,
                                 const ProblemSpecP& module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
  m_lb = scinew particleExtractLabel();
  M_lb = scinew MPMLabel();

  m_lb->lastWriteTimeLabel        = VarLabel::create("lastWriteTime_partE", max_vartype::getTypeDescription());
  m_lb->filePointerLabel          = VarLabel::create("filePointer", ParticleVariable< FILE* >::getTypeDescription() );
  m_lb->filePointerLabel_preReloc = VarLabel::create("filePointer+", ParticleVariable< FILE* >::getTypeDescription() );

}

//__________________________________
particleExtract::~particleExtract()
{
  DOUTR(dout_OTF_partEx, " Doing: destructor particleExtract" );

  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy( m_lb->lastWriteTimeLabel );
  VarLabel::destroy( m_lb->filePointerLabel );
  VarLabel::destroy( m_lb->filePointerLabel_preReloc );
  delete m_lb;
  delete M_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void particleExtract::problemSetup(const ProblemSpecP& ,
                                   const ProblemSpecP& ,
                                   GridP& grid,
                                   std::vector<std::vector<const VarLabel* > > &PState,
                                   std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{

  DOUTR(dout_OTF_partEx, "Doing particleExtract::problemSetup");

  if(m_module_spec->findBlock("material") ){
    d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }
  else {
    throw ProblemSetupException("ERROR:AnalysisModule:particleExtract: Missing <material> tag. \n", __FILE__, __LINE__);
  }

  vector<int> m;
  m.push_back( d_matl->getDWIndex() );

  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", m_analysisFreq);
  m_module_spec->require("timeStart",         d_startTime);
  m_module_spec->require("timeStop",          d_stopTime);

  m_module_spec->require("colorThreshold",    d_colorThreshold);
  //__________________________________
  //  Read in variables label names
  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("particleExtract: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }
  map<string,string> attribute;
  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {
    var_spec->getAttributes(attribute);
    string     name  = attribute["label"];
    VarLabel* label  = VarLabel::find( name, "ERROR  particleExtract::problemSetup:analyze");

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
  if(d_startTime > d_stopTime){
    throw ProblemSetupException("\n ERROR:particleExtract: startTime > stopTime. \n", __FILE__, __LINE__);
  }

  // Tell MPM that these variable need to be relocated
  int matl = d_matl->getDWIndex();
  PState[matl].push_back(         m_lb->filePointerLabel);
  PState_preReloc[matl].push_back(m_lb->filePointerLabel_preReloc);

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
void particleExtract::scheduleInitialize(SchedulerP   & sched,
                                         const LevelP & level)
{
  int L_indx = level->getIndex();
  if(!doMPMOnLevel(L_indx,level->getGrid()->numLevels())){
    return;
  }

  printSchedule(level, dout_OTF_partEx, "particleExtract::scheduleInitialize");

  Task* t = scinew Task("particleExtract::initialize",
                  this, &particleExtract::initialize);

  t->computes( m_lb->lastWriteTimeLabel );
  t->computes( m_lb->filePointerLabel ) ;
  sched->addTask( t, level->eachPatch(), d_matl_set );
}
//______________________________________________________________________
void particleExtract::initialize(const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset *,
                                 DataWarehouse        *,
                                 DataWarehouse        * new_dw)
{
  double tminus = d_startTime - 1.0/m_analysisFreq;
  new_dw->put( max_vartype( tminus ), m_lb->lastWriteTimeLabel );

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dout_OTF_partEx, "Doing particleExtract::initialize");

    ParticleVariable<FILE*> myFiles;
    int indx = d_matl->getDWIndex();
    ParticleSubset* pset = new_dw->getParticleSubset( indx, patch );
    new_dw->allocateAndPut( myFiles, m_lb->filePointerLabel, pset );

    for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
      particleIndex idx = *iter;
      myFiles[idx] = nullptr;
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
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:particleExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}
//______________________________________________________________________
void particleExtract::scheduleRestartInitialize(SchedulerP   & sched,
                                                const LevelP & level)
{
  scheduleInitialize( sched, level);
}

//______________________________________________________________________
void particleExtract::scheduleDoAnalysis_preReloc(SchedulerP& sched,
                                                  const LevelP& level)
{
  int L_indx = level->getIndex();
  if(!doMPMOnLevel(L_indx,level->getGrid()->numLevels())){
    return;
  }

  printSchedule(level, dout_OTF_partEx, "particleExtract::scheduleDoAnalysis_preReloc");

  Task* t = scinew Task("particleExtract::doAnalysis_preReloc",
                   this,&particleExtract::doAnalysis_preReloc);

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.  Put it here so it will be registered during a restart
  sched->overrideVariableBehavior("filePointer", false, false, false, true, true);

  t->needsLabel( Task::OldDW,  m_lb->filePointerLabel, m_gn, 0 );
  t->computes( m_lb->filePointerLabel_preReloc  );

  sched->addTask(t, level->eachPatch(),  d_matl_set);
}
//______________________________________________________________________
//
void particleExtract::doAnalysis_preReloc(const ProcessorGroup* pg,
                                          const PatchSubset* patches,
                                          const MaterialSubset*,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dout_OTF_partEx,"Doing particleExtract::doAnalysis_preReloc");

    int indx = d_matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(indx, patch);
    constParticleVariable<FILE*> myFiles;
    ParticleVariable<FILE*>      myFiles_preReloc;

    new_dw->allocateAndPut( myFiles_preReloc, m_lb->filePointerLabel_preReloc, pset );

    // Only transfer forward myFiles if they exist.  The filePointerLabel is NOT
    // saved in the checkpoints and so you can't get it from the old_dw.
    if( old_dw->exists( m_lb->filePointerLabel, indx, patch ) ){
      old_dw->get( myFiles,  m_lb->filePointerLabel,  pset );

      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        myFiles_preReloc[idx] = myFiles[idx];
      }
    }
    else{

      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
        myFiles_preReloc[idx] = nullptr;
      }
    }
  }
}

//______________________________________________________________________
//
void particleExtract::scheduleDoAnalysis(SchedulerP& sched,
                                         const LevelP& level)
{
  // only schedule task on the finest level
  int L_indx = level->getIndex();
  if(!doMPMOnLevel(L_indx,level->getGrid()->numLevels())){
    return;
  }

  printSchedule(level, dout_OTF_partEx,"particleExtract::scheduleDoAnalysis");

  Task* t = scinew Task("particleExtract::doAnalysis",
                   this,&particleExtract::doAnalysis);

  sched_TimeVars( t, level, m_lb->lastWriteTimeLabel, true );


  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == nullptr){
      string name = d_varLabels[i]->getName();
      throw InternalError("particleExtract: scheduleDoAnalysis label not found: "
                          + name , __FILE__, __LINE__);
    }
    t->needsLabel(Task::NewDW,d_varLabels[i], m_gn, 0);
  }

  t->needsLabel( Task::NewDW,  M_lb->pXLabel,           m_gn );
  t->needsLabel( Task::NewDW,  M_lb->pParticleIDLabel,  m_gn );
  t->needsLabel( Task::NewDW,  M_lb->pColorLabel,       m_gn );
  t->needsLabel( Task::NewDW,  m_lb->filePointerLabel,  m_gn );
  t->modifies( m_lb->filePointerLabel );

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void
particleExtract::doAnalysis( const ProcessorGroup * pg,
                             const PatchSubset    * patches,
                             const MaterialSubset *,
                             DataWarehouse        * old_dw,
                             DataWarehouse        * new_dw )
{
  const Level* level = getLevel(patches);

  timeVars tv;

  getTimeVars( old_dw, level, m_lb->lastWriteTimeLabel, tv );
  putTimeVars( new_dw, m_lb->lastWriteTimeLabel, tv );

  if( tv.isItTime == false ){
    return;
  }

  //__________________________________
  //
  const string udaDir = m_output->getOutputLocation();
  const string path = "particleExtract/L-" + to_string( level->getIndex() );

  createDirectory( 0777, udaDir, path );

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int proc = m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);

    DOUTR(dbg_OTF_partEx, "   working on patch " << patch->getID() << " which is on proc " << proc );
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myRank() ){

      printTask(patches, patch, dout_OTF_partEx,"Doing particleExtract::doAnalysis");

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

      int NGP = 0;
      int indx = d_matl->getDWIndex();
      ParticleSubset* pset = new_dw->getParticleSubset(indx, patch,
                                                 m_gn, NGP, M_lb->pXLabel);

      // additional particle data
      new_dw->get(pid,    M_lb->pParticleIDLabel, pset);
      new_dw->get(px,     M_lb->pXLabel,          pset);
      new_dw->get(pColor, M_lb->pColorLabel,      pset);

      // file pointers
      ParticleVariable<FILE*>myFiles;
      new_dw->getModifiable( myFiles,    m_lb->filePointerLabel, pset );

      //__________________________________
      //  Put particle data into arrays <double,int,....>_data
      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {

        // bulletproofing
        if(d_varLabels[i] == nullptr){
          string name = d_varLabels[i]->getName();
          throw InternalError("particleExtract: analyze label (" + name +" ) not found: ", __FILE__, __LINE__);
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

      //__________________________________
      // loop over the particle
      for (ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if (pColor[idx] > d_colorThreshold){

          ostringstream fname;
          fname <<udaDir << "/"<<  path << "/" <<pid[idx];
          string filename = fname.str();

          // open the file
          FILE *fp = nullptr;
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
          fprintf(fp,    "%E   %E   %E   %E",tv.now, px[idx].x(),px[idx].y(),px[idx].z());

           // WARNING If you change the order that these are written
           // out you must also change the order that the header is
           // written

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
          //  HACK:  Close each file and set the fp to nullptr
          //  Remove this hack once we figure out how to use
          //  particle relocation to move file pointers between
          //  patches.
          fclose(fp);
          myFiles[idx] = nullptr;
        }
      }  // loop over particles
    }  // time to write data
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
    const string name = d_varLabels[i]->getName();

    if( subtype->getType() == TypeDescription::int_type    ||
        subtype->getType() == TypeDescription::double_type ){
      fprintf(fp,"     %s", name.c_str());
    }

    if(subtype->getType() == TypeDescription::Vector){
      string name = d_varLabels[i]->getName();
      fprintf(fp,"     %s.x      %s.y      %s.z", name.c_str(),name.c_str(),name.c_str());
    }

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
//
bool
particleExtract::doMPMOnLevel(int level, int numLevels)
{
  return ( level == numLevels - 1 );
}
