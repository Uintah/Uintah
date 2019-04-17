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

#include <CCA/Components/OnTheFlyAnalysis/findFragments.h>

#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <dirent.h>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;


Dout dbg_OTF_FS("findFragments", false);
//______________________________________________________________________
findFragments::findFragments(ProblemSpecP     & module_spec,
                             SimulationStateP & sharedState,
                             Output           * dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set     = 0;
  d_lb           = scinew findFragmentsLabel();
  d_lb->prevAnalysisTimeLabel = VarLabel::create( "prevAnalysisTime", max_vartype::getTypeDescription() );
}

//__________________________________
findFragments::~findFragments()
{
  DOUT(dbg_OTF_FS , " Doing: destorying findFragments " );
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy( d_lb->prevAnalysisTimeLabel );
  VarLabel::destroy( d_lb->fileVarsStructLabel );
  delete d_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void findFragments::problemSetup(const ProblemSpecP & prob_spec,
                                 const ProblemSpecP & ,
                                 GridP              & grid,
                                 SimulationStateP   & sharedState)
{
  DOUT(dbg_OTF_FS , "Doing problemSetup \t\t\t\tfindFragments" );

  if(!d_dataArchiver){
    throw InternalError("findFragments:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_analysisFreq);
  d_prob_spec->require("timeStart",         d_startTime);
  d_prob_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("findFragments: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  // find the material.
  //  <material>   atmosphere </material>

  const Material* matl = nullptr;
  matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");

  int defaultMatl = matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->add( defaultMatl );
  d_matl_set->addReference();

  d_matl_subSet = d_matl_set->getUnion();

  //__________________________________
  //  Read in variables label names
  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {

    map<string,string> attribute;
    var_spec->getAttributes(attribute);

    string name     = attribute["label"];
    VarLabel* label = VarLabel::find(name);

    if(label == nullptr){
      throw ProblemSetupException("findFragments: analyze label not found: "+ name , __FILE__, __LINE__);
    }

    const Uintah::TypeDescription* td      = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    //__________________________________
    // bulletproofing

    // Only NC Variables, Doubles and Vectors
    if(td->getType()      != TypeDescription::NCVariable  &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::Vector  ){

      ostringstream warn;
      warn << "ERROR:AnalysisModule:findFragments: ("<<label->getName() << " "
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    d_varLabels.push_back(label);
  }
}

//______________________________________________________________________
void findFragments::scheduleInitialize( SchedulerP   & sched,
                                        const LevelP & level)
{
  printSchedule(level,dbg_OTF_FS, "findFragments::scheduleInitialize " );

  Task* t = scinew Task("findFragments::initialize",
                  this, &findFragments::initialize);

  t->computes(d_lb->prevAnalysisTimeLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void findFragments::initialize(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        *,
                               DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, dbg_OTF_FS,"Doing  planeExtract::initialize" );

    double tminus = -1.0/d_analysisFreq;
    new_dw->put(max_vartype(tminus), d_lb->prevAnalysisTimeLabel);


    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:findFragments  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}

//______________________________________________________________________
void findFragments::scheduleRestartInitialize(SchedulerP   & sched,
                                              const LevelP & level)
{
  scheduleInitialize( sched, level);
}

//______________________________________________________________________
void findFragments::scheduleDoAnalysis(SchedulerP   & sched,
                                       const LevelP & level)
{
  printSchedule(level,dbg_OTF_FS, "findFragments::scheduleDoAnalysis");

  Task* t = scinew Task("findFragments::doAnalysis",
                   this,&findFragments::doAnalysis);

  t->requires(Task::OldDW, d_lb->prevAnalysisTimeLabel);
  Ghost::GhostType gac = Ghost::AroundCells;

  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    t->requires(Task::NewDW, d_varLabels[i], d_matl_subSet, gac, 1);
  }

  t->computes(d_lb->prevAnalysisTimeLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void findFragments::doAnalysis(const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw)
{
  UintahParallelComponent * DA = dynamic_cast<UintahParallelComponent*>( d_dataArchiver );
  LoadBalancerPort        * lb = dynamic_cast<LoadBalancerPort*>( DA->getPort("load balancer") );

  const Level* level = getLevel(patches);

  // the user may want to restart from an uda that wasn't using the DA module
  // This logic allows that.
  max_vartype writeTime;
  double prevAnalysisTime = 0;
  if( old_dw->exists( d_lb->prevAnalysisTimeLabel ) ){
    old_dw->get(writeTime, d_lb->prevAnalysisTimeLabel);
    prevAnalysisTime = writeTime;
  }

  double now = d_sharedState->getElapsedTime();

  if(now < d_startTime || now > d_stopTime){
    new_dw->put(max_vartype(prevAnalysisTime), d_lb->prevAnalysisTimeLabel);
    return;
  }

  double nextWriteTime = prevAnalysisTime + 1.0/d_analysisFreq;

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, dbg_OTF_FS,"Doing findFragments::doAnalysis" );

    int proc = lb->getPatchwiseProcessorAssignment(patch);

    prevAnalysisTime = now;

    new_dw->put(max_vartype(prevAnalysisTime), d_lb->prevAnalysisTimeLabel);
  }  // patches
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void findFragments::createFile( const string   & filename,
                                const VarLabel * varLabel,
                                const int matl,
                                FILE           *& fp )
{
  // if the file already exists then exit.
  ifstream doExists( filename.c_str() );
  if( doExists ){
    fp = fopen(filename.c_str(), "a");
    return;
  }

  fp = fopen(filename.c_str(), "w");

  if (!fp){
    perror("Error opening file:");
    throw InternalError("\nERROR:dataAnalysisModule:findFragments:  failed opening file"+filename,__FILE__, __LINE__);
  }

  //__________________________________
  //Write out the header
  fprintf(fp,"# X      Y      Z ");


  const Uintah::TypeDescription* td = varLabel->typeDescription();
  const Uintah::TypeDescription* subtype = td->getSubType();
  string labelName = varLabel->getName();

  switch( subtype->getType( )) {

    case Uintah::TypeDescription::double_type:
      fprintf(fp,"     %s(%i)", labelName.c_str(), matl);
      break;

    case Uintah::TypeDescription::Vector:
      fprintf(fp,"     %s(%i).x      %s(%i).y      %s(%i).z", labelName.c_str(), matl, labelName.c_str(), matl, labelName.c_str(), matl);
      break;

    default:
      throw InternalError("findFragments: invalid data type", __FILE__, __LINE__);
  }

  fprintf(fp,"\n");
  fflush(fp);

  cout << Parallel::getMPIRank() << " findFragments:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure
//
void
findFragments::createDirectory(string      & planeName,
                               string      & timestep,
                               const double  now,
                               string      & levelIndex)
{
  DIR *check = opendir(planeName.c_str());
  if ( check == nullptr ) {
    cout << Parallel::getMPIRank() << " findFragments:Making directory " << planeName << endl;
    MKDIR( planeName.c_str(), 0777 );
  } else {
    closedir(check);
  }

  // timestep
  string path = planeName + "/" + timestep;
  check = opendir( path.c_str() );

  if ( check == nullptr ) {
    cout << Parallel::getMPIRank() << " findFragments:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );

    // write out physical time
    string filename = planeName + "/" + timestep + "/physicalTime";
    FILE *fp;
    fp = fopen( filename.c_str(), "w" );
    fprintf( fp, "%16.15E\n",now);
    fclose(fp);

  } else {
    closedir(check);
  }

  // level index
  path = planeName + "/" + timestep + "/" + levelIndex;
  check = opendir( path.c_str() );

  if ( check == nullptr ) {
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
