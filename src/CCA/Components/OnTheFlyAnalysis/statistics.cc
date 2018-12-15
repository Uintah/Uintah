/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/statistics.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/visit_defs.h>

#include <iostream>
#include <cstdio>
#include <iomanip>

//______________________________________________________________________
//    TO DO:
//   Each variable needs to keep track of the timestep.  The user can add
//   a variable in a checkpoint.
//______________________________________________________________________
//

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("STATISTICS_DOING_COUT", false);
static DebugStream cout_dbg("STATISTICS_DBG_COUT", false);

//______________________________________________________________________
statistics::statistics( const ProcessorGroup* myworld,
                        const MaterialManagerP materialManager,
                        const ProblemSpecP& module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matlSet     = 0;
  d_stopTime    = DBL_MAX;
  d_monitorCell = IntVector(0,0,0);
  d_doHigherOrderStats = false;

  // Reynolds Shear Stress related
  d_RS_matl     = -9;
  d_computeReynoldsStress = false;

  required = false;
}

//__________________________________
statistics::~statistics()
{
  cout_doing << " Doing: destorying statistics " << endl;
  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }

  // delete each Qstats label
  for (unsigned int i =0 ; i < d_Qstats.size(); i++) {
    Qstats& Q = d_Qstats[i];
    VarLabel::destroy( Q.Qsum_Label );
    VarLabel::destroy( Q.Qsum2_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qmean2_Label );
    VarLabel::destroy( Q.Qvariance_Label );

    if( d_doHigherOrderStats ){
      VarLabel::destroy( Q.Qsum3_Label );
      VarLabel::destroy( Q.Qmean3_Label );
      VarLabel::destroy( Q.Qsum4_Label );
      VarLabel::destroy( Q.Qmean4_Label );
    }
  }

  if ( d_computeReynoldsStress ){
    VarLabel::destroy( d_velPrime_Label );
    VarLabel::destroy( d_velSum_Label  );
    VarLabel::destroy( d_velMean_Label  );
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void statistics::problemSetup(const ProblemSpecP &,
                              const ProblemSpecP & restart_prob_spec,
                              GridP & grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tstatistics" << endl;

  int numMatls  = m_materialManager->getNumMatls();

  //__________________________________
  //  Read in timing information
  m_module_spec->require("timeStart",  d_startTime);
  m_module_spec->require("timeStop",   d_stopTime);
  
  // Start time < stop time
  if(d_startTime > d_stopTime ){
    throw ProblemSetupException("\n ERROR:statistics: startTime > stopTime. \n", __FILE__, __LINE__);
  }
  
  // debugging
  m_module_spec->get("monitorCell",    d_monitorCell);


  //__________________________________
  //  read in when each variable started 
  string comment = "__________________________________\n"
                   "\tIf you want to overide the value of\n \t  startTimeTimestep\n \t  startTimeTimestepReynoldsStress\n"
                   "\tsee checkpoints/t*****/timestep.xml\n"
                   "\t__________________________________";
  m_module_spec->addComment( comment ) ;

  
  //__________________________________
  // find the material to extract data from.  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>

  Material* matl = nullptr;

  if(m_module_spec->findBlock("material") ){
    matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  } 
  else if (m_module_spec->findBlock("materialIndex") ){
    int indx;
    m_module_spec->get("materialIndex", indx);
    matl = m_materialManager->getMaterial(indx);
  } 
  else {
    matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  proc0cout << "__________________________________ Data Analysis module: statistics" << endl;
  m_module_spec->get("computeHigherOrderStats", d_doHigherOrderStats );
  if (d_doHigherOrderStats){

    proc0cout << "         Computing 2nd, 3rd and 4th order statistics for all of the variables listed"<< endl;
  } 
  else {
    proc0cout << "         Computing 2nd order statistics for all of the variables listed"<< endl;
  }

  //__________________________________
  //  Read in variables label names

   ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("statistics: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {
    map<string,string> attribute;
    var_spec->getAttributes(attribute);

    //__________________________________
    //  Read in the optional material index from the variables that may be different
    //  from the default index and construct the material set
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }

    // bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("statistics: problemSetup: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);


    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == nullptr){
      throw ProblemSetupException("statistics label not found: " + name , __FILE__, __LINE__);
    }

    //__________________________________
    // Only CCVariable Doubles and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType() != TypeDescription::CCVariable  ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::Vector ) ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:statisticst: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    Qstats Q;
    Q.matl    = matl;
    Q.Q_Label = label;
    Q.subtype = subtype;
    Q.computeRstess = false;
    Q.initializeTimestep();          // initialize the start timestep = 0;

    Q.Qsum_Label      = VarLabel::create( "sum_" + name,      td);
    Q.Qsum2_Label     = VarLabel::create( "sum2_" + name,     td);
    Q.Qmean_Label     = VarLabel::create( "mean_" + name,     td);
    Q.Qmean2_Label    = VarLabel::create( "mean2_" + name,    td);
    Q.Qvariance_Label = VarLabel::create( "variance_" + name, td);

    if( d_doHigherOrderStats ){
      Q.Qsum3_Label     = VarLabel::create( "sum3_" + name,      td);
      Q.Qmean3_Label    = VarLabel::create( "mean3_" + name,     td);
      Q.Qskewness_Label = VarLabel::create( "skewness_" + name,  td);

      Q.Qsum4_Label     = VarLabel::create( "sum4_" + name,      td);
      Q.Qmean4_Label    = VarLabel::create( "mean4_" + name,     td);
      Q.Qkurtosis_Label = VarLabel::create( "kurtosis_" + name,  td);
    }

    //__________________________________
    //  computeReynoldsStress with this Var?
    if (attribute["computeReynoldsStress"].empty() == false){
      d_computeReynoldsStress = true;
      Q.computeRstess    = true;
      d_RS_matl = matl;
      proc0cout << "         Computing uv_prime, uw_prime, vw_prime using ("<< name << ")" << endl;
    }

    //__________________________________
    // keep track of which summation variables
    // have been initialized.  A user can
    // add a variable on a restart.  Default is false.
    Q.isInitialized[lowOrder]     = false;
    Q.isInitialized[highOrder]    = false;
    d_isReynoldsStressInitialized = false;

    d_Qstats.push_back( Q );

    //__________________________________
    //  bulletproofing
    std::string variance = "variance_"+ name;
    std::string skew     = "skewness_"+ name;
    std::string kurtosis = "kurtosis_"+ name;
    ostringstream mesg;
    mesg << "";
    if( !m_output->isLabelSaved( variance ) ){
      mesg << variance;
    }
    if( !m_output->isLabelSaved( skew )  && d_doHigherOrderStats){
      mesg << " " << skew;
    }
    if( !m_output->isLabelSaved( kurtosis ) && d_doHigherOrderStats){
      mesg << " " << kurtosis;
    }

    if( mesg.str() != "" ){
      ostringstream warn;
      warn << "WARNING:  You've activated the DataAnalysis:statistics module but your not saving the variable(s) ("
           << mesg.str() << ")";
      proc0cout << warn.str() << endl;
    }
  }


  //__________________________________
  //  computeReynoldsStress with this Var?
  if ( d_computeReynoldsStress){
    const TypeDescription* td = CCVariable<Vector>::getTypeDescription();
    d_velPrime_Label = VarLabel::create( "uv_vw_wu_prime", td);
    d_velSum_Label   = VarLabel::create( "sum_uv_vw_wu",   td);
    d_velMean_Label  = VarLabel::create( "mean_uv_vw_wu",  td);
  }

  //__________________________________
  //  On restart read the starttimestep for each variable from checkpoing/t***/timestep.xml
  if(restart_prob_spec){ 
    ProblemSpecP da_rs_ps = restart_prob_spec->findBlock("DataAnalysisRestart");
    
    ProblemSpecP stat_ps = da_rs_ps->findBlockWithAttributeValue("Module", "name", "statistics");
    ProblemSpecP st_ps   = stat_ps->findBlock("StartTimestep");
    
    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];
      int timestep;
      st_ps->require( Q.Q_Label->getName().c_str(), timestep  );
      Q.setStart(timestep);
      proc0cout <<  "         " << Q.Q_Label->getName() << "\t\t startTimestep: " << timestep << endl;                   
      
    }
  }

  //__________________________________
  //  create the matl set
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  d_matlSet = scinew MaterialSet();
  d_matlSet->addAll(m);
  d_matlSet->addReference();
  d_matSubSet = d_matlSet->getUnion();
  proc0cout << "__________________________________ Data Analysis module: statistics" << endl;
  
#ifdef HAVE_VISIT
  static bool initialized = false;

  if( m_application->getVisIt() && !initialized ) {
    required = true;

    initialized = true;
  }
#endif
}

//______________________________________________________________________
void statistics::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleInitialize" );

  Task* t = scinew Task("statistics::initialize",
                   this,&statistics::initialize);

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    const Qstats Q = d_Qstats[i];

    t->computes ( Q.Qsum_Label );
    t->computes ( Q.Qsum2_Label );

    if( d_doHigherOrderStats ){
      t->computes ( Q.Qsum3_Label );
      t->computes ( Q.Qsum4_Label );
    }
  }

  //__________________________________
  //  For Reynolds Stress components
  if( d_computeReynoldsStress ){
    t->computes ( d_velSum_Label );
  }

  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
//
void statistics::initialize(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing statistics::initialize");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroSums<double>( new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroSums<Vector>( new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("statistics: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstat

    //__________________________________
    //
    if( d_computeReynoldsStress && !d_isReynoldsStressInitialized ){
      proc0cout << "    Statistics: initializing summation variables needed for Reynolds Stress calculation" << endl;
      allocateAndZero<Vector>( new_dw, d_velSum_Label,  d_RS_matl, patch );
    }
  }  // pathes
}

//______________________________________________________________________
// This allows the user to restart from an uda in which this module was
// turned off.  Only execute task if the labels were NOT in the checkpoint
void statistics::scheduleRestartInitialize(SchedulerP& sched,
                                           const LevelP& level)
{

  printSchedule( level,cout_doing,"statistics::scheduleRestartInitialize" );

  DataWarehouse* new_dw = sched->getLastDW();

  // Find the first patch on this level that this mpi rank owns.
  const Uintah::PatchSet* const ps =
    sched->getLoadBalancer()->getPerProcessorPatchSet(level);
  int rank = Parallel::getMPIRank();
  const PatchSubset* myPatches = ps->getSubset(rank);
  const Patch* firstPatch = myPatches->get(0);

  Task* t = scinew Task("statistics::restartInitialize",
                   this,&statistics::restartInitialize);

  bool addTask = false;

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];

    // Do the summation Variables exist in checkpoint
    //              low order
    if (new_dw->exists( Q.Qsum_Label, Q.matl, firstPatch) ){
      Q.isInitialized[lowOrder] = true;
      d_Qstats[i].isInitialized[lowOrder] = true;
    }


    //              high order
    if( d_doHigherOrderStats ){
      if ( new_dw->exists( Q.Qsum3_Label, Q.matl, firstPatch) ){
        Q.isInitialized[highOrder] = true;
        d_Qstats[i].isInitialized[highOrder] = true;
      }
    }

    // if the Q.sum was not in previous checkpoint compute it
    if( !Q.isInitialized[lowOrder] ){
      t->computes ( Q.Qsum_Label );
      t->computes ( Q.Qsum2_Label );
      addTask = true;
      proc0cout << "    Statistics: Adding lowOrder computes for " << Q.Q_Label->getName() << endl;
    }

    if( d_doHigherOrderStats && !Q.isInitialized[highOrder] ){
      t->computes ( Q.Qsum3_Label );
      t->computes ( Q.Qsum4_Label );
      addTask = true;
      proc0cout << "    Statistics: Adding highOrder computes for " << Q.Q_Label->getName() << endl;
    }
  }

  //__________________________________
  //  Reynolds stress
  // Do the summation Variables exist in checkpoint
  if(d_computeReynoldsStress ){
    if (new_dw->exists( d_velSum_Label, d_RS_matl, firstPatch) ){
      d_isReynoldsStressInitialized = true;
    } else {
      t->computes ( d_velSum_Label );
      addTask = true;
      proc0cout << "    Statistics: Adding computes for Reynolds Stress (u'v', u'w', w'u') terms "  << endl;
    }
  }

  // only add task if a variable was not found in old_dw
  if ( addTask ){
    sched->addTask(t, level->eachPatch(), d_matlSet);
  }
}


//______________________________________________________________________
//
void statistics::restartInitialize(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing statistics::restartInitialize");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats Q = d_Qstats[i];
      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroSums<double>( new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroSums<Vector>( new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("statistics: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstat

    //__________________________________
    //
    if ( d_computeReynoldsStress && !d_isReynoldsStressInitialized ){
      proc0cout << "    Statistics: initializing summation variables needed for Reynolds Stress calculation" << endl;
      allocateAndZero<Vector>( new_dw, d_velSum_Label,  d_RS_matl, patch );
    }

  }  // pathes
}

//______________________________________________________________________
//
void
statistics::restartInitialize()
{
}

//______________________________________________________________________
//  output the starting timestep for each variable
//  The user can turn add variables on restarts
void
statistics::outputProblemSpec( ProblemSpecP& root_ps)
{
  if( root_ps == nullptr ) {
    throw InternalError("ERROR: DataAnalysis Module:statistics::outputProblemSpec:  ProblemSpecP is nullptr", __FILE__, __LINE__);
  }

  ProblemSpecP da_ps = root_ps->appendChild("DataAnalysisRestart");

  ProblemSpecP m_ps = da_ps->appendChild("Module");
  m_ps->setAttribute( "name","statistics" );
  ProblemSpecP st_ps = m_ps->appendChild("StartTimestep");

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];
    st_ps->appendElement( Q.Q_Label->getName().c_str(), Q.getStart() );
  }
}

//______________________________________________________________________
void statistics::scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleDoAnalysis" );

  Task* t = scinew Task("statistics::doAnalysis",
                   this,&statistics::doAnalysis);

  t->requires(Task::OldDW, m_timeStepLabel);
  t->requires(Task::OldDW, m_simulationTimeLabel);
  
  Ghost::GhostType  gn  = Ghost::None;
  
  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];

    // define the matl subset for this variable
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( Q.matl );
    matSubSet->addReference();

    //__________________________________
    //  Lower order statistics
    t->requires( Task::NewDW, Q.Q_Label,     matSubSet, gn, 0 );
    t->requires( Task::OldDW, Q.Qsum_Label,  matSubSet, gn, 0 );
    t->requires( Task::OldDW, Q.Qsum2_Label, matSubSet, gn, 0 );

#ifdef HAVE_VISIT
    if( required )
    {
      t->requires( Task::OldDW, Q.Qmean_Label,     matSubSet, gn, 0 );
      t->requires( Task::OldDW, Q.Qmean2_Label,    matSubSet, gn, 0 );
      t->requires( Task::OldDW, Q.Qvariance_Label, matSubSet, gn, 0 );
    }
#endif
    
    t->computes ( Q.Qsum_Label,       matSubSet );
    t->computes ( Q.Qsum2_Label,      matSubSet );
    t->computes ( Q.Qmean_Label,      matSubSet );
    t->computes ( Q.Qmean2_Label,     matSubSet );
    t->computes ( Q.Qvariance_Label,  matSubSet );

    //__________________________________
    // Higher order statistics
    if( d_doHigherOrderStats ){

      t->requires( Task::OldDW, Q.Qsum3_Label, matSubSet, gn, 0 );
      t->requires( Task::OldDW, Q.Qsum4_Label, matSubSet, gn, 0 );

#ifdef HAVE_VISIT
      if( required )
      {
        t->requires( Task::OldDW, Q.Qmean3_Label,    matSubSet, gn, 0 );
        t->requires( Task::OldDW, Q.Qmean4_Label,    matSubSet, gn, 0 );
        t->requires( Task::OldDW, Q.Qskewness_Label, matSubSet, gn, 0 );
        t->requires( Task::OldDW, Q.Qkurtosis_Label, matSubSet, gn, 0 );
      }
#endif

      t->computes ( Q.Qsum3_Label,     matSubSet );
      t->computes ( Q.Qsum4_Label,     matSubSet );
      t->computes ( Q.Qmean3_Label,    matSubSet );
      t->computes ( Q.Qmean4_Label,    matSubSet );
      t->computes ( Q.Qskewness_Label, matSubSet );
      t->computes ( Q.Qkurtosis_Label, matSubSet );
    }
    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
//    Q.print();
  }

  //__________________________________
  //  Reynolds Stress Terms
  if ( d_computeReynoldsStress ){
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( d_RS_matl );
    matSubSet->addReference();

    t->requires( Task::OldDW, d_velSum_Label,  matSubSet, gn, 0 );

#ifdef HAVE_VISIT
    if( required )
    {
      t->requires( Task::OldDW, d_velPrime_Label, matSubSet, gn, 0 );
      t->requires( Task::OldDW, d_velMean_Label,  matSubSet, gn, 0 );
    }
#endif

    t->computes ( d_velPrime_Label,  matSubSet );
    t->computes ( d_velSum_Label,    matSubSet );
    t->computes ( d_velMean_Label,   matSubSet );
    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }

  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
// Compute the statistics for each variable the user requested
void statistics::doAnalysis(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* ,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing statistics::doAnalysis");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          computeStatsWrapper< double >(old_dw, new_dw, patches, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          computeStatsWrapper< Vector >(old_dw, new_dw, patches,  patch, Q);

          computeReynoldsStressWrapper( old_dw, new_dw, patches,  patch, Q);

          break;
        }
        default: {
          throw InternalError("statistics: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // qstats loop
  }  // patches
}




//______________________________________________________________________
//  computeStatsWrapper:
template <class T>
void statistics::computeStatsWrapper( DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      const PatchSubset* patches,
                                      const Patch*    patch,
                                      Qstats& Q)
{
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){

//    proc0cout << " IGNORING------------DataAnalysis: Statistics" << endl;
    allocateAndZeroStats<T>( new_dw, patch, Q);
    carryForwardSums( old_dw, new_dw, patches, Q );
  }
  else {
//    proc0cout << " Computing------------DataAnalysis: Statistics" << endl;

    computeStats< T >(old_dw, new_dw, patch, Q);
  }
}



//______________________________________________________________________
//
template <class T>
void statistics::computeStats( DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const Patch*    patch,
                               Qstats& Q)
{
  const int matl = Q.matl;

  constCCVariable<T> Qvar;
  constCCVariable<T> Qsum_old;
  constCCVariable<T> Qsum2_old;

  Ghost::GhostType  gn  = Ghost::None;
  new_dw->get ( Qvar,      Q.Q_Label,      matl, patch, gn, 0 );
  old_dw->get ( Qsum_old,  Q.Qsum_Label,   matl, patch, gn, 0 );
  old_dw->get ( Qsum2_old, Q.Qsum2_Label,  matl, patch, gn, 0 );

  CCVariable< T > Qsum;
  CCVariable< T > Qsum2;
  CCVariable< T > Qmean;
  CCVariable< T > Qmean2;
  CCVariable< T > Qvariance;

  new_dw->allocateAndPut( Qsum,      Q.Qsum_Label,      matl, patch );
  new_dw->allocateAndPut( Qsum2,     Q.Qsum2_Label,     matl, patch );
  new_dw->allocateAndPut( Qmean,     Q.Qmean_Label,     matl, patch );
  new_dw->allocateAndPut( Qmean2,    Q.Qmean2_Label,    matl, patch );
  new_dw->allocateAndPut( Qvariance, Q.Qvariance_Label, matl, patch );

  timeStep_vartype timeStep_var;      
  old_dw->get(timeStep_var, m_timeStepLabel);
  int ts = timeStep_var;

  Q.setStart(ts);
  int Q_ts = Q.getStart();
  int timestep = ts - Q_ts + 1;

  T nTimesteps(timestep);

  //__________________________________
  //  Lower order stats  1st and 2nd
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    T me = Qvar[c];     // for readability
    Qsum[c]    = me + Qsum_old[c];
    Qmean[c]   = Qsum[c]/nTimesteps;

    Qsum2[c]   = me * me + Qsum2_old[c];
    Qmean2[c]  = Qsum2[c]/nTimesteps;

    Qvariance[c] = Qmean2[c] - Qmean[c] * Qmean[c];

#if 0
    //__________________________________
    //  debugging
    if ( c == d_monitorCell ){
      cout << "  stats:  " << d_monitorCell <<  setw(10)<< Q.Q_Label->getName() << " nTimestep: " << nTimesteps
           <<"\t timestep " << ts
           << " d_startTimeStep: " << d_startTimeTimestep
           <<"\t Q_var: " << me
           <<"\t Qsum: "  << Qsum[c]
           <<"\t Qmean: " << Qmean[c] << endl;
    }
#endif

  }

  //__________________________________
  //  Higher order stats  3rd and 4th
  if( d_doHigherOrderStats ){

    constCCVariable<T> Qsum3_old;
    constCCVariable<T> Qsum4_old;

    old_dw->get ( Qsum3_old, Q.Qsum3_Label, matl, patch, gn, 0 );
    old_dw->get ( Qsum4_old, Q.Qsum4_Label, matl, patch, gn, 0 );

    CCVariable< T > Qsum3;
    CCVariable< T > Qsum4;
    CCVariable< T > Qmean3;
    CCVariable< T > Qmean4;

    CCVariable< T > Qskewness;
    CCVariable< T > Qkurtosis;
    new_dw->allocateAndPut( Qsum3,     Q.Qsum3_Label,     matl, patch );
    new_dw->allocateAndPut( Qsum4,     Q.Qsum4_Label,     matl, patch );
    new_dw->allocateAndPut( Qmean3,    Q.Qmean3_Label,    matl, patch );
    new_dw->allocateAndPut( Qmean4,    Q.Qmean4_Label,    matl, patch );
    new_dw->allocateAndPut( Qskewness, Q.Qskewness_Label, matl, patch );
    new_dw->allocateAndPut( Qkurtosis, Q.Qkurtosis_Label, matl, patch );

    for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      T me = Qvar[c];     // for readability
      T me2 = me * me;
      T Qbar = Qmean[c];
      T Qbar2 = Qbar * Qbar;
      T Qbar3 = Qbar * Qbar2;
      T Qbar4 = Qbar2 * Qbar2;

      // skewness
      Qsum3[c]  = me * me2 + Qsum3_old[c];
      Qmean3[c] = Qsum3[c]/nTimesteps;

      Qskewness[c] = Qmean3[c] - Qbar3 - 3 * Qvariance[c] * Qbar;

      // kurtosis
      Qsum4[c]  = me2 * me2 + Qsum4_old[c];
      Qmean4[c] = Qsum4[c]/nTimesteps;

      Qkurtosis[c] = Qmean4[c] - Qbar4
                   - 6 * Qvariance[c] * Qbar2
                   - 4 * Qskewness[c] * Qbar;
    }
  }
}
//______________________________________________________________________
//  computeReynoldsStressWrapper:
void statistics::computeReynoldsStressWrapper( DataWarehouse* old_dw,
                                               DataWarehouse* new_dw,
                                               const PatchSubset* patches,
                                               const Patch*    patch,
                                               Qstats& Q)
{
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){

//    proc0cout << " IGNORING------------statistics::computeReynoldsStress" << endl;
    // define the matl subset for this variable
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( Q.matl );
    matSubSet->addReference();

    new_dw->transferFrom(    old_dw, d_velSum_Label,   patches, matSubSet );
    allocateAndZero<Vector>( new_dw, d_velPrime_Label, d_RS_matl, patch );
    allocateAndZero<Vector>( new_dw, d_velMean_Label,  d_RS_matl, patch );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }else {
//    proc0cout << " Computing------------statistics::computeReynoldsStress" << endl;
    computeReynoldsStress( old_dw, new_dw,patch, Q);
  }
}

//______________________________________________________________________
//  Computes u'v', u'w', v'w'
void statistics::computeReynoldsStress( DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        const Patch* patch,
                                        Qstats& Q)
{
  if ( Q.computeRstess == false ){
    return;
  }

  const int matl = Q.matl;
  constCCVariable<Vector> vel;
  constCCVariable<Vector> vel_mean;
  constCCVariable<Vector> Qsum_old;

  Ghost::GhostType  gn  = Ghost::None;
  new_dw->get ( vel,       Q.Q_Label,        matl, patch, gn, 0 );
  new_dw->get ( vel_mean,  Q.Qmean_Label,    matl, patch, gn, 0 );
  old_dw->get ( Qsum_old,  d_velSum_Label,   matl, patch, gn, 0 );

  CCVariable< Vector > Qsum;
  CCVariable< Vector > Qmean;
  CCVariable< Vector > uv_vw_wu;

  new_dw->allocateAndPut( Qsum,      d_velSum_Label,    matl, patch );
  new_dw->allocateAndPut( Qmean,     d_velMean_Label,   matl, patch );
  new_dw->allocateAndPut( uv_vw_wu,  d_velPrime_Label,  matl, patch );
  
  timeStep_vartype timeStep_var;      
  old_dw->get(timeStep_var, m_timeStepLabel);
  int ts = timeStep_var;

  Q.setStart(ts);
  int Q_ts = Q.getStart();
  int timestep = ts - Q_ts + 1;

  Vector nTimesteps(timestep);

  //__________________________________
  // UV_prime(i) = mean(U .* V) - mean(U) .* mean(V);
  // UW_prime(i) = mean(U .* W) - mean(U) .* mean(W);
  // VW_prime(i) = mean(V .* W) - mean(V) .* mean(W)

  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    Vector me  = Multiply(vel[c], vel[c] );   /// u = uv, vw, wu
    Qsum[c]    = me + Qsum_old[c];
    Qmean[c]   = Qsum[c]/nTimesteps;


    uv_vw_wu[c] = Qmean[c] - Multiply(vel_mean[c],vel_mean[c]);
#if 0
    //__________________________________
    //  debugging
    if ( c == d_monitorCell ){
      cout << "  ReynoldsStress stats:  \n \t \t"<< d_monitorCell << " nTimestep: " << nTimesteps.x()
           <<  " timestep " << ts
           << " d_startTimeTimestepReynoldsStress: " << d_startTimeTimestepReynoldsStress
           <<"\n \t \t"<<Q.Q_Label->getName()<< ": " << vel[c]<< " vel_CC_mean: " << vel_mean[c]
           <<"\n \t \tuv_vw_wu: " << me << ",  uv_vw_wu_sum: " << Qsum[c]<< ",  uv_vw_wu_mean: " << Qmean[c]
           <<"\n \t \tuv_vw_wu_prime: " <<  uv_vw_wu[c] << endl;
    }
#endif
  }
}

//______________________________________________________________________
//  allocateAndZero  statistics variables
template <class T>
void statistics::allocateAndZeroStats( DataWarehouse* new_dw,
                                      const Patch* patch,
                                      const Qstats& Q )
{
  int matl = Q.matl;
  allocateAndZero<T>( new_dw, Q.Qvariance_Label,  matl, patch );
  allocateAndZero<T>( new_dw, Q.Qmean_Label,      matl, patch );

  if( d_doHigherOrderStats ){
    allocateAndZero<T>( new_dw, Q.Qskewness_Label, matl, patch );
    allocateAndZero<T>( new_dw, Q.Qkurtosis_Label, matl, patch );
  }

}

//______________________________________________________________________
//  allocateAndZero  summation variables
template <class T>
void statistics::allocateAndZeroSums( DataWarehouse* new_dw,
                                      const Patch* patch,
                                      Qstats& Q )
{
  int matl = Q.matl;
  if ( !Q.isInitialized[lowOrder] ){
    allocateAndZero<T>( new_dw, Q.Qsum_Label,  matl, patch );
    allocateAndZero<T>( new_dw, Q.Qsum2_Label, matl, patch );
//    proc0cout << "    Statistics: " << Q.Q_Label->getName() << " initializing low order sums on patch: " << patch->getID()<<endl;
  }

  if( d_doHigherOrderStats && !Q.isInitialized[highOrder] ){
    allocateAndZero<T>( new_dw, Q.Qsum3_Label, matl, patch );
    allocateAndZero<T>( new_dw, Q.Qsum4_Label, matl, patch );
//    proc0cout << "    Statistics: " << Q.Q_Label->getName() << " initializing high order sums on patch: " << patch->getID() << endl;
  }
}

//______________________________________________________________________
//  allocateAndZero
template <class T>
void statistics::allocateAndZero( DataWarehouse* new_dw,
                                  const VarLabel* label,
                                  const int       matl,
                                  const Patch*    patch )
{
  CCVariable<T> Q;
  new_dw->allocateAndPut( Q, label, matl, patch );
  T zero(0.0);
  Q.initialize( zero );
}


//______________________________________________________________________
//  carryForward  summation variables
void statistics::carryForwardSums( DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   const PatchSubset* patches,
                                   const Qstats& Q )
{
    // define the matl subset for this variable
  MaterialSubset* matSubSet = scinew MaterialSubset();
  matSubSet->add( Q.matl );
  matSubSet->addReference();

  new_dw->transferFrom(old_dw, Q.Qsum_Label,  patches, matSubSet  );
  new_dw->transferFrom(old_dw, Q.Qsum2_Label, patches, matSubSet );

  if( d_doHigherOrderStats ){
    new_dw->transferFrom(old_dw, Q.Qsum3_Label, patches, matSubSet );
    new_dw->transferFrom(old_dw, Q.Qsum4_Label, patches, matSubSet );
  }

  if(matSubSet && matSubSet->removeReference()){
    delete matSubSet;
  }
}
