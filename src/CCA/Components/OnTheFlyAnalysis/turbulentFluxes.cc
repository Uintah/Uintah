/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/turbulentFluxes.h>

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

#include <iostream>
#include <cstdio>
#include <iomanip>

#define DEBUG

//______________________________________________________________________
//    TO DO:
//  DOUT streams
//  update ups_spec
//  Add bulletproofing.  A velocity label must be added.
//  destroy varLabels
//  Run through fsanitize for memory leaks
//  Polish
// Testing:
//   single patch single processor
//   multiprocessor
//   Can user add variables after a restart
//   Correctness
//   Correctness after starttime != 0
//
//______________________________________________________________________
//

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("turbulentFluxes_DOING_COUT", false);
static DebugStream cout_dbg("turbulentFluxes_DBG_COUT", false);

//______________________________________________________________________
//
turbulentFluxes::turbulentFluxes( const ProcessorGroup  * myworld,
                                  const MaterialManagerP materialManager,
                                  const ProblemSpecP    & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_stopTime    = DBL_MAX;
  m_monitorCell = IntVector(0,0,0);
}

//__________________________________
turbulentFluxes::~turbulentFluxes()
{
  cout_doing << " Doing: destorying turbulentFluxes " << endl;
  if(m_matlSet && m_matlSet->removeReference()) {
    delete m_matlSet;
  }

  // delete each Qvar label
  for (unsigned int i =0 ; i < m_Qvars.size(); i++) {
    Qvar_ptr Q = m_Qvars[i];
    VarLabel::destroy( Q->Qsum_Label );
    VarLabel::destroy( Q->Qmean_Label );

    if( Q->matlSubset && Q->matlSubset->removeReference()){
      delete Q->matlSubset;
    }
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void turbulentFluxes::problemSetup(const ProblemSpecP &,
                                   const ProblemSpecP & restart_prob_spec,
                                   GridP & grid,
                                   std::vector<std::vector<const VarLabel* > > &PState,
                                   std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tturbulentFluxes" << endl;

  int numMatls  = m_materialManager->getNumMatls();

  //__________________________________
  //  Read in timing information
  m_module_spec->require("timeStart",  d_startTime);
  m_module_spec->require("timeStop",   d_stopTime);

  // Start time < stop time
  if( d_startTime > d_stopTime ){
    throw ProblemSetupException("\n ERROR:turbulentFluxes: startTime > stopTime. \n", __FILE__, __LINE__);
  }

  // debugging
  m_module_spec->get("monitorCell",    m_monitorCell);


  //__________________________________
  //  read in when each variable started
  string comment = "__________________________________\n"
                   "\tIf you want to overide the value of\n \t  startTimeTimestep\n \t  startTimeTimestepReynoldsStress\n"
                   "\tsee checkpoints/t*****/timestep.xml\n"
                   "\t__________________________________";
  m_module_spec->addComment( comment ) ;


  //__________________________________
  // find the material to extract data from.
  //  <material>   atmosphere </material>

  Material* matl = nullptr;

  if(m_module_spec->findBlock("material") ){
    matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }

  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  proc0cout << "__________________________________ Data Analysis module: turbulentFluxes" << endl;
  proc0cout << "         Computing the turbulentFluxes for all of the variables listed"<< endl;

  //__________________________________
  //  Read in variables label names

   ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("turbulentFluxes: Couldn't find <Variables> tag", __FILE__, __LINE__);
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
      throw ProblemSetupException("turbulentFluxes: problemSetup: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);


    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);

    if( label == nullptr ){
      throw ProblemSetupException("turbulentFluxes label not found: " + name , __FILE__, __LINE__);
    }

    //__________________________________
    // Only CCVariable doubles and Vectors for now
    const TypeDescription * td     = label->typeDescription();
    const TypeDescription * td_V   = CCVariable<Vector>::getTypeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType() != TypeDescription::CCVariable  ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::Vector ) ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:turbulentFluxest: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the required labels for this variable
    auto Q     = make_shared< Qvar >(matl);
    Q->Label   = label;
    Q->subtype = subtype;
    Q->initializeTimestep();          // initialize the start timestep = 0;

    Q->Qsum_Label          = VarLabel::create( "sum_"    + name,        td );
    Q->Q2sum_Label         = VarLabel::create( "sum_"    + name + "2",  td );
    Q->Qmean_Label         = VarLabel::create( "mean_"   + name,        td );
    Q->Q2mean_Label        = VarLabel::create( "mean_"   + name + "2",  td );
    Q->Qu_Qv_Qw_sum_Label  = VarLabel::create( "sum_Qu_Qv_Qw_"  + name, td_V );
    Q->Qu_Qv_Qw_mean_Label = VarLabel::create( "mean_Qu_Qv_Qw_" + name, td_V );

    Q->variance_Label      = VarLabel::create( "variance_"   + name,   td);
    Q->covariance_Label    = VarLabel::create( "covariance_" + name, td_V);

    //__________________________________
    //  computeReynoldsStress with this Var?
    if ( attribute["fluidVelocityLabel"].empty() == false ){
      m_velVar = make_shared< velocityVar >(matl);
      proc0cout << "         Computing uv_prime, uw_prime, vw_prime using ("<< name << ")" << endl;
    }

    //__________________________________
    // keep track of which summation variables
    // have been initialized.  A user can
    // add a variable on a restart.  Default is false.
    Q->isInitialized     = false;
    m_Qvars.push_back( Q );

    //__________________________________
    //  bulletproofing
    std::string variance = "variance_"+ name;
    ostringstream mesg;
    mesg << "";
    if( !m_output->isLabelSaved( variance ) ){
      mesg << variance;
    }

    if( mesg.str() != "" ){
      ostringstream warn;
      warn << "WARNING:  You've activated the DataAnalysis:turbulentFluxes module but your not saving the variable(s) ("
           << mesg.str() << ")";
      proc0cout << warn.str() << endl;
    }
  }

  //__________________________________
  //  On restart read the starttimestep for each variable from checkpoing/t***/timestep.xml
  if(restart_prob_spec){
    ProblemSpecP da_rs_ps = restart_prob_spec->findBlock("DataAnalysisRestart");

    ProblemSpecP stat_ps = da_rs_ps->findBlockWithAttributeValue("Module", "name", "turbulentFluxes");
    ProblemSpecP st_ps   = stat_ps->findBlock("StartTimestep");

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      Qvar_ptr Q = m_Qvars[i];
      int timestep;
      st_ps->require( Q->Label->getName().c_str(), timestep  );
      Q->setStart(timestep);
      proc0cout <<  "         " << Q->Label->getName() << "\t\t startTimestep: " << timestep << endl;

    }
  }

  //__________________________________
  //  create the matl set
  m_matlSet = scinew MaterialSet();
  m_matlSet->addAll_unique(m);
  m_matlSet->addReference();
  proc0cout << "__________________________________ Data Analysis module: turbulentFluxes" << endl;
}

//______________________________________________________________________
//
void turbulentFluxes::scheduleInitialize( SchedulerP   & sched,
                                          const LevelP & level)
{
  printSchedule( level,cout_doing,"turbulentFluxes::scheduleInitialize" );

  Task* t = scinew Task("turbulentFluxes::initialize",
                   this,&turbulentFluxes::initialize);

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];
    t->computes ( Q->Qsum_Label );
    t->computes ( Q->Q2sum_Label );
    t->computes ( Q->Qu_Qv_Qw_sum_Label );

    t->computes ( Q->Qmean_Label );
    t->computes ( Q->Q2mean_Label );
    t->computes ( Q->Qu_Qv_Qw_mean_Label );
  }

  sched->addTask(t, level->eachPatch(), m_matlSet);
}

//______________________________________________________________________
//
void turbulentFluxes::initialize( const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        *,
                                  DataWarehouse        * new_dw)
{

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing turbulentFluxes::initialize");

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      const Qvar_ptr Q = m_Qvars[i];

      switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroSums<double>(  new_dw, patch, Q);
          allocateAndZeroMeans<double>( new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroSums<Vector>(  new_dw, patch, Q);
          allocateAndZeroMeans<Vector>( new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("turbulentFluxes: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qvars

  }  // patches
}

//______________________________________________________________________
// This allows the user to restart from an uda in which this module was
// turned off.  Only execute task if the labels were NOT in the checkpoint
void turbulentFluxes::scheduleRestartInitialize( SchedulerP    & sched,
                                                 const LevelP  & level)
{

  printSchedule( level,cout_doing,"turbulentFluxes::scheduleRestartInitialize" );

  DataWarehouse* new_dw = sched->getLastDW();

  // Find the first patch on this level that this mpi rank owns.
  const Uintah::PatchSet* const ps =
    sched->getLoadBalancer()->getPerProcessorPatchSet(level);
  int rank = Parallel::getMPIRank();
  const PatchSubset* myPatches = ps->getSubset(rank);
  const Patch* firstPatch = myPatches->get(0);

  Task* t = scinew Task("turbulentFluxes::restartInitialize",
                   this,&turbulentFluxes::restartInitialize);

  bool addTask = false;

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];

    // Do the summation Variables exist in checkpoint
    if ( new_dw->exists( Q->Qsum_Label, Q->matl, firstPatch) ){
      Q->isInitialized = true;
      m_Qvars[i]->isInitialized = true;
    }

    // if the Q->sum was not in previous checkpoint compute it
    if( !Q->isInitialized ){
      t->computes ( Q->Qsum_Label );
      addTask = true;
      proc0cout << "    turbulentFluxes: Adding lowOrder computes for " << Q->Label->getName() << endl;
    }
  }

  // only add task if a variable was not found in old_dw
  if ( addTask ){
    sched->addTask(t, level->eachPatch(), m_matlSet);
  }
}


//______________________________________________________________________
//
void turbulentFluxes::restartInitialize( const ProcessorGroup  *,
                                         const PatchSubset     * patches,
                                         const MaterialSubset  *,
                                         DataWarehouse         *,
                                         DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing turbulentFluxes::restartInitialize");

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      Qvar_ptr Q = m_Qvars[i];
      switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroSums<double>( new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroSums<Vector>( new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("turbulentFluxes: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstatx
  }  // pathes
}

//______________________________________________________________________
//
void
turbulentFluxes::restartInitialize()
{
}

//______________________________________________________________________
//  output the starting timestep for each variable
//  The user can turn add variables on restarts
void
turbulentFluxes::outputProblemSpec( ProblemSpecP& root_ps)
{
  if( root_ps == nullptr ) {
    throw InternalError("ERROR: DataAnalysis Module:turbulentFluxes::outputProblemSpec:  ProblemSpecP is nullptr", __FILE__, __LINE__);
  }

  ProblemSpecP da_ps = root_ps->appendChild("DataAnalysisRestart");

  ProblemSpecP m_ps = da_ps->appendChild("Module");
  m_ps->setAttribute( "name","turbulentFluxes" );
  ProblemSpecP st_ps = m_ps->appendChild("StartTimestep");

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];
    st_ps->appendElement( Q->Label->getName().c_str(), Q->getStart() );
  }
}

//______________________________________________________________________
//
void turbulentFluxes::scheduleDoAnalysis( SchedulerP    & sched,
                                          const LevelP  & level)
{
  sched_Q_mean( sched, level);

  sched_turbFluxes( sched, level);

}


//______________________________________________________________________
//
void turbulentFluxes::sched_Q_mean( SchedulerP   & sched,
                                    const LevelP & level)
{
  Task* t = scinew Task( "turbulentFluxes::task_Q_mean",
                    this,&turbulentFluxes::task_Q_mean );

  printSchedule(level,cout_doing,"turbulentFluxes::sched_Q_mean");

  t->requires(Task::OldDW, m_timeStepLabel);
  t->requires(Task::OldDW, m_simulationTimeLabel);

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    Qvar_ptr Q = m_Qvars[i];

    const MaterialSubset* matl = Q->matlSubset;

    //__________________________________
    //
    t->requires( Task::NewDW, Q->Label,               matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->Qsum_Label,          matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->Q2sum_Label,         matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->Qu_Qv_Qw_sum_Label,  matl, m_gn, 0 );

    t->computes( Q->Qsum_Label,          matl );
    t->computes( Q->Q2sum_Label,         matl );
    t->computes( Q->Qmean_Label,         matl );
    t->computes( Q->Q2mean_Label,        matl );
    t->computes( Q->Qu_Qv_Qw_sum_Label,  matl );
    t->computes( Q->Qu_Qv_Qw_mean_Label, matl );
//    Q->print();
  }
  sched->addTask( t, level->eachPatch() , m_matlSet );
}


//______________________________________________________________________
//
void turbulentFluxes::task_Q_mean( const ProcessorGroup * ,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * ,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw )
{

  //__________________________________
  //  not yet time
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){
    carryForward( old_dw, new_dw, patches );
    return;
  }

  //__________________________________
  //  Time to compute something
  for( auto p=0;p<patches->size();p++ ){

    const Patch* patch = patches->get(p);
    printTask(patches, patch, cout_doing, "Doing turbulentFluxes::Q_mean");

    constCCVariable<Vector> vel;
    new_dw->get ( vel, m_velVar->Label, m_velVar->matl, patch, m_gn, 0 );

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      Qvar_ptr Q = m_Qvars[i];

      switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          Q_mean< double >( old_dw, new_dw, patch, Q, vel );
          break;
        }
        case TypeDescription::Vector: {             // Vector
          Q_mean< Vector >( old_dw, new_dw, patch, Q, vel );

          break;
        }
        default: {
          throw InternalError("turbulentFluxes: invalid data type", __FILE__, __LINE__);
        }
      }
    }  //  loop Qvars
  }  // loop patches
}

//______________________________________________________________________
//
template<class T>
void turbulentFluxes::Q_mean( DataWarehouse * old_dw,
                              DataWarehouse * new_dw,
                              const Patch   * patch,
                              const Qvar_ptr  Q,
                              constCCVariable<Vector> vel )

{
  const int matl = Q->matl;

  constCCVariable<T> Qvar;
  constCCVariable<T> Qsum_old;
  constCCVariable<T> Q2sum_old;
  constCCVariable<Vector> Qu_Qv_Qw_sum_old;

  new_dw->get ( Qvar,             Q->Label,              matl, patch, m_gn, 0 );
  old_dw->get ( Qsum_old,         Q->Qsum_Label,         matl, patch, m_gn, 0 );
  old_dw->get ( Q2sum_old,        Q->Q2sum_Label,        matl, patch, m_gn, 0 );
  old_dw->get ( Qu_Qv_Qw_sum_old, Q->Qu_Qv_Qw_sum_Label, matl, patch, m_gn, 0 );

  CCVariable< T > Qsum;
  CCVariable< T > Q2sum;
  CCVariable< T > Qmean;
  CCVariable< T > Q2mean;
  CCVariable< Vector > Qu_Qv_Qw_sum;
  CCVariable< Vector > Qu_Qv_Qw_mean;

  new_dw->allocateAndPut( Qsum,         Q->Qsum_Label,          matl, patch );
  new_dw->allocateAndPut( Q2sum,        Q->Q2sum_Label,         matl, patch );
  new_dw->allocateAndPut( Qmean ,       Q->Qmean_Label,         matl, patch );
  new_dw->allocateAndPut( Q2mean ,      Q->Q2mean_Label,        matl, patch );
  new_dw->allocateAndPut( Qu_Qv_Qw_sum, Q->Qu_Qv_Qw_sum_Label,  matl, patch );
  new_dw->allocateAndPut( Qu_Qv_Qw_mean, Q->Qu_Qv_Qw_mean_Label,matl, patch );

  timeStep_vartype timeStep_var;
  old_dw->get(timeStep_var, m_timeStepLabel);
  int ts = timeStep_var;

  Q->setStart(ts);
  int Q_ts = Q->getStart();
  int timestep = ts - Q_ts + 1;

  T nTimesteps(timestep);

  //__________________________________
  //
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    T var = Qvar[c];     // for readability
    Qsum[c]   = var + Qsum_old[c];
    Qmean[c]  = Qsum[c]/nTimesteps;

    Q2sum[c]  = multiply_Q_Q(var, var) + Q2sum_old[c];
    Q2mean[c] = Q2sum[c]/nTimesteps;

    Qu_Qv_Qw_sum[c]  = multiply_Q_Vel(var, vel[c]) + Qu_Qv_Qw_sum_old[c];
    Qu_Qv_Qw_mean[c] = Qu_Qv_Qw_sum[c]/nTimesteps;


#ifdef DEBUG
    //__________________________________
    //  debugging
    if ( c == m_monitorCell ){
      cout << "  turbulentFluxes:  " << m_monitorCell <<  setw(10)<< Q->Label->getName() << " nTimestep: " << nTimesteps
           <<"\t timestep " << ts
           <<"\t Q_var: " << var
           <<"\t Qsum: "  << Qsum[c]
           <<"\t Qmean: " << Qmean[c]
           <<"\t Q2sum: " << Q2sum[c]
           <<"\t Q2mean: " << Q2mean[c]<< endl;
    }
#endif

  }

}


//______________________________________________________________________
//
void turbulentFluxes::sched_turbFluxes( SchedulerP   & sched,
                                        const LevelP & level)
{
  Task* t = scinew Task( "turbulentFluxes::task_turbFluxes",
                    this,&turbulentFluxes::task_turbFluxes );

  printSchedule(level,cout_doing,"turbulentFluxes::sched_turbFluxes");

  t->requires(Task::OldDW, m_timeStepLabel);
  t->requires(Task::OldDW, m_simulationTimeLabel);

  //__________________________________
  //
  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    Qvar_ptr Q = m_Qvars[i];

    const MaterialSubset* matl = Q->matlSubset;

    t->requires( Task::NewDW, Q->Label,                matl, m_gn, 0 );
    t->requires( Task::NewDW, Q->Qmean_Label,          matl, m_gn, 0 );
    t->requires( Task::NewDW, Q->Q2mean_Label,         matl, m_gn, 0 );
    t->requires( Task::NewDW, Q->Qu_Qv_Qw_mean_Label,  matl, m_gn, 0 );

    t->computes ( Q->variance_Label,   matl );
    t->computes ( Q->covariance_Label, matl );
//    Q->print();
  }
  sched->addTask( t, level->eachPatch() , m_matlSet );
}

//______________________________________________________________________
//
//______________________________________________________________________
//
void turbulentFluxes::task_turbFluxes( const ProcessorGroup  * ,
                                       const PatchSubset     * patches,
                                       const MaterialSubset  * ,
                                       DataWarehouse         * old_dw,
                                       DataWarehouse         * new_dw)
{

  //__________________________________
  //
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){
    return;
  }

  //__________________________________
  //
  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, cout_doing, "Doing turbulentFluxes::variance");

    constCCVariable<Vector> velMean;
    new_dw->get ( velMean, m_velVar->Qmean_Label, m_velVar->matl, patch, m_gn, 0 );

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      Qvar_ptr Q = m_Qvars[i];

        switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          turbFluxes< double >( new_dw, patch, Q, velMean );
          break;
        }
        case TypeDescription::Vector: {             // Vector
          turbFluxes< Vector >( new_dw, patch, Q, velMean );

          break;
        }
        default: {
          throw InternalError("turbulentFluxes: invalid data type", __FILE__, __LINE__);
        }
      }

    }  // loop Qvars
  }  // patches
}


//______________________________________________________________________
//
template<class T>
void turbulentFluxes::turbFluxes( DataWarehouse * new_dw,
                                  const Patch   * patch,
                                  const Qvar_ptr  Q,
                                  constCCVariable<Vector> velMean )
{
  const int matl = Q->matl;

  constCCVariable< T > Qmean;
  constCCVariable< T > Q2mean;
  constCCVariable< Vector > Qu_Qv_Qw_mean;

  new_dw->get ( Qmean,         Q->Qmean_Label,          matl, patch, m_gn, 0 );
  new_dw->get ( Q2mean,        Q->Q2mean_Label,         matl, patch, m_gn, 0 );
  new_dw->get ( Qu_Qv_Qw_mean, Q->Qu_Qv_Qw_mean_Label,  matl, patch, m_gn, 0 );

  CCVariable< T >     variance;
  CCVariable< Vector > covariance;
  new_dw->allocateAndPut( variance,   Q->variance_Label,  matl, patch );
  new_dw->allocateAndPut( covariance, Q->covariance_Label,matl, patch );

  //__________________________________
  //
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    const T Q_mean = Qmean[c];     // for readability
    const Vector mean_Qu_Qv_Qw = Qu_Qv_Qw_mean[c];
    const Vector vel_mean      = velMean[c];

    /* variance of Q <double>  = double  (mean(Q * Q)  - mean(Q)*mean(Q)

       variance of Q <Vector>  = Vector( (mean(Q.x * u)  - mean(Q.x)*mean(u)
                                         (mean(Q.y * v)  - mean(Q.y)*mean(v)
                                         (mean(Q.z * w)  - mean(Q.z)*mean(w)
    */
    variance[c]   = Q2mean[c] - multiply_Q_Q( Q_mean, Q_mean );


    /* covariance of a Q <double>  = Vector( (mean(Q * u)  - mean(Q)*mean(u)
                                             (mean(Q * v)  - mean(Q)*mean(v)
                                             (mean(Q * w)  - mean(Q)*mean(w)

     covariance of Q <Vector> =      Vector( (mean(Q.x * v)  - mean(Q.x)*mean(v)
                                             (mean(Q.y * w)  - mean(Q.y)*mean(w)
                                             (mean(Q.z * u)  - mean(Q.z)*mean(u)
    */
    covariance[c] = mean_Qu_Qv_Qw - multiply_Q_Vel( Q_mean, vel_mean );



#ifdef DEBUG
    //__________________________________
    //  debugging
    if ( c == m_monitorCell ){
      cout << "  TurbFluctuations:  " << m_monitorCell <<  setw(10)<< Q->Label->getName()
           <<"\t variance: "  << variance[c]
           <<"\t covariance: " << covariance[c] << endl;
    }
#endif
  }
}

//______________________________________________________________________
//                    UTILITIES
//______________________________________________________________________
//  allocateAndZero averages variables
template <class T>
void turbulentFluxes::allocateAndZeroMeans( DataWarehouse * new_dw,
                                            const Patch   * patch,
                                            Qvar_ptr Q )
{
  int matl = Q->matl;
  allocateAndZero<T>(      new_dw, Q->Qmean_Label,          matl, patch );
  allocateAndZero<T>(      new_dw, Q->Q2mean_Label,         matl, patch );
  allocateAndZero<Vector>( new_dw, Q->Qu_Qv_Qw_mean_Label, matl, patch );
}

//______________________________________________________________________
//  allocateAndZero  summation variables
template <class T>
void turbulentFluxes::allocateAndZeroSums( DataWarehouse* new_dw,
                                           const Patch  * patch,
                                           Qvar_ptr Q )
{
  int matl = Q->matl;
  if ( !Q->isInitialized ){
    allocateAndZero<T>(      new_dw, Q->Qsum_Label,           matl, patch );
    allocateAndZero<T>(      new_dw, Q->Q2sum_Label,          matl, patch );
    allocateAndZero<Vector>( new_dw, Q->Qu_Qv_Qw_sum_Label,  matl, patch );
//    proc0cout << "    turbulentFluxes: " << Q->Label->getName() << " initializing low order sums on patch: " << patch->getID()<<endl;
  }
}

//______________________________________________________________________
//  allocateAndZero
template <class T>
void turbulentFluxes::allocateAndZero( DataWarehouse  * new_dw,
                                       const VarLabel * label,
                                       const int        matl,
                                       const Patch    * patch )
{
  CCVariable<T> Q;
  new_dw->allocateAndPut( Q, label, matl, patch );
  T zero(0.0);
  Q.initialize( zero );
}


//______________________________________________________________________
//  carryForward  variables from old_dw to new_dw
void turbulentFluxes::carryForward( DataWarehouse     * old_dw,
                                    DataWarehouse     * new_dw,
                                    const PatchSubset * patches )
{
  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];

    new_dw->transferFrom(old_dw, Q->Qsum_Label,           patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->Q2sum_Label,          patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->Qu_Qv_Qw_sum_Label,   patches, Q->matlSubset );

    new_dw->transferFrom(old_dw, Q->Qmean_Label,          patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->Q2mean_Label,         patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->Qu_Qv_Qw_mean_Label,  patches, Q->matlSubset );

    new_dw->transferFrom(old_dw, Q->variance_Label,       patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->covariance_Label,     patches, Q->matlSubset );
  }
}
