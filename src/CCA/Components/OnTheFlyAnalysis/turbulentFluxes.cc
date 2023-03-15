/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <cstdio>
#include <iomanip>

#define DEBUG

// return std:cout if rank 0 and the x == Y
#define proc0cout_cmp(X,Y) if( isProc0_macro && X == Y) std::cout

//______________________________________________________________________
//    TO DO:

// This will not work with Adaptive Mesh Refinement. The computed quantities would need
// to be set to zero and a separate firstSumTimestep would we needed for each variable and
// patch.  It doesn't make sense at this point.  Mesh refinement is fine.
// Testing:

//   multiprocessor/ multipatch
//   Correctness
//   Correctness after starttime != 0
//
//______________________________________________________________________
//

using namespace Uintah;
using namespace std;

Dout dbg_OTF_TF("turbulentFluxes", "OnTheFlyAnalysis", "Task scheduling and execution.", false);

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

//______________________________________________________________________
//
turbulentFluxes::~turbulentFluxes()
{
  DOUTR(dbg_OTF_TF, " Doing: destructor turbulentFluxes " )


  if(m_matlSet && m_matlSet->removeReference()) {
    delete m_matlSet;
  }

  // delete each Qvar label
  for (unsigned int i =0 ; i < m_Qvars.size(); i++) {

    Qvar_ptr Q = m_Qvars[i];
    VarLabel::destroy( Q->Qsum_Label );
    VarLabel::destroy( Q->Q2sum_Label );
    VarLabel::destroy( Q->Qmean_Label );
    VarLabel::destroy( Q->Q2mean_Label );
    VarLabel::destroy( Q->Qu_Qv_Qw_sum_Label );
    VarLabel::destroy( Q->Qu_Qv_Qw_mean_Label );

    VarLabel::destroy( Q->variance_Label );
    VarLabel::destroy( Q->covariance_Label );

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
  DOUTR(dbg_OTF_TF, "turbulentFluxes::problemSetup" );

  int numMatls  = m_materialManager->getNumMatls();

  //__________________________________
  //  Bulletproofing, no adaptivity
  bool amr = m_application->isDynamicRegridding();

  if( amr){
    std::string err;
    err = "\nERROR:AnalysisModule:turbulentFluxes: This module will not work with";
    err += " an adaptive mesh.  It does work a multi level static grid.";
    throw ProblemSetupException( err, __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  m_module_spec->require("timeStart",  d_startTime);
  m_module_spec->require("timeStop",   d_stopTime);

  // Start time < stop time
  if( d_startTime > d_stopTime ){
    throw ProblemSetupException("ERROR:AnalysisModule:turbulentFluxes: startTime > stopTime. \n", __FILE__, __LINE__);
  }

  // debugging
  m_module_spec->get("monitorCell",    m_monitorCell);


  //__________________________________
  //  read in when each variable started
  string comment = "__________________________________\n"
                   "\tIf you want to overide the value of\n \t  firstSumTimestep\n"
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
  else {
    throw ProblemSetupException("ERROR:AnalysisModule:turbulentFluxes: Missing <material> tag. \n", __FILE__, __LINE__);
  }

  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  proc0cout << "__________________________________ Data Analysis module: turbulentFluxes" << endl;
  proc0cout << "         Computing the turbulentFluxes and intermediate values:"<< endl;

  //__________________________________
  //  Read in variables label names

   ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("ERROR:AnalysisModule:turbulentFluxes: Couldn't find <Variables> tag", __FILE__, __LINE__);
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
      throw ProblemSetupException("ERROR:AnalysisModule:turbulentFluxes: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);

    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find( name, "ERROR turbulentFluxes::problemSetup <analyze>");

    //__________________________________
    // Only CCVariable < doubles > and < Vectors > for now
    const TypeDescription * td     = label->typeDescription();
    const TypeDescription * td_V   = CCVariable<Vector>::getTypeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType() != TypeDescription::CCVariable  ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::Vector ) ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:turbulentFluxes: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the required labels
    auto Q     = make_shared< Qvar >(matl);
    Q->Label   = label;
    Q->subtype = subtype;

    Q->Qsum_Label          = createVarLabel( "sum_"    + name,        td );
    Q->Q2sum_Label         = createVarLabel( "sum_"    + name + "2",  td );
    Q->Qmean_Label         = createVarLabel( "mean_"   + name,        td );
    Q->Q2mean_Label        = createVarLabel( "mean_"   + name + "2",  td );
    Q->Qu_Qv_Qw_sum_Label  = createVarLabel( "sum_Qu_Qv_Qw_"  + name, td_V );
    Q->Qu_Qv_Qw_mean_Label = createVarLabel( "mean_Qu_Qv_Qw_" + name, td_V );

    Q->variance_Label      = createVarLabel( "variance_"   + name,   td);
    Q->covariance_Label    = createVarLabel( "covariance_" + name, td_V);

    //__________________________________
    //  Is this the fluid velocity label?
    if ( attribute["fluidVelocityLabel"].empty() == false ){
      m_velVar=Q;
    }

    //__________________________________
    // keep track of which summation variables
    // have been initialized.  A user can
    // add a variable on a restart.  Default is false.
    Q->isInitialized = false;
    m_Qvars.push_back( Q );

  }

  //__________________________________
  //  On restart read the firstSumTimestep for each variable from checkpoing/t***/timestep.xml
  if( restart_prob_spec ){

    ProblemSpecP da_rs_ps = restart_prob_spec->findBlock("DataAnalysisRestart");
    if( da_rs_ps ){
      ProblemSpecP stat_ps = da_rs_ps->findBlockWithAttributeValue("Module", "name", "turbulentFluxes");
      ProblemSpecP vars_ps = stat_ps->findBlock("variables");

      for(ProblemSpecP n = vars_ps->getFirstChild(); n != nullptr; n=n->getNextSibling()) {

        if(n->getNodeName() == "variable") {
          map<string,string> attributes;
          n->getAttributes( attributes );
          const string varname = attributes["name"];

          for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
            Qvar_ptr Q = m_Qvars[i];

            const string name =  Q->Label->getName();

            if( varname == name ){
              Q->firstSumTimestep = std::stoi( attributes[ "firstSumTimestep" ] );
              Q->isStatEnabled    = std::stoi( attributes[ "isStatEnabled" ] );


              break;
            }
          }  // loop over Qvars
        }  // variable
      }  // loop over children
    }
  }  //restart


  //__________________________________
  //  Warnings

  ostringstream warn;
  warn << "";

  for ( unsigned int i =0 ; i < m_VarLabelNames.size(); i++ ) {
    std::string name = m_VarLabelNames[i];

    if( !m_output->isLabelSaved( name ) ){
      warn << "\t" << name << "\n";
    }
  }

  if( warn.str() != "" ){
    warn << "WARNING:  You've activated the DataAnalysis:turbulentFluxes module but your not saving the variable(s) (\n"
         << warn.str() << ")";
    proc0cout << warn.str() << endl;
  }


  //__________________________________
  //  bulletproofing
  if ( !m_velVar ){
    throw ProblemSetupException("ERROR:turbulentFluxes: A label for the fluid velocity [fluidVelocityLabel] was not found.", __FILE__, __LINE__);
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
  printSchedule( level,dbg_OTF_TF,"turbulentFluxes::scheduleInitialize" );

  Task* t = scinew Task("turbulentFluxes::initialize",
                   this,&turbulentFluxes::initialize);

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];
    const MaterialSubset* matl = Q->matlSubset;

    t->computes ( Q->Qsum_Label,          matl );
    t->computes ( Q->Q2sum_Label,         matl );
    t->computes ( Q->Qu_Qv_Qw_sum_Label,  matl );

    t->computes ( Q->Qmean_Label,         matl );
    t->computes ( Q->Q2mean_Label,        matl );
    t->computes ( Q->Qu_Qv_Qw_mean_Label, matl );

    t->computes ( Q->variance_Label,      matl );
    t->computes ( Q->covariance_Label,    matl );
  }

  sched->addTask(t, level->eachPatch(), m_matlSet);
}

//______________________________________________________________________
//
//   Zero out computed quantites if they don't exist in the new_dw
void turbulentFluxes::initialize( const ProcessorGroup *,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        *,
                                  DataWarehouse        * new_dw)
{

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg_OTF_TF, "Doing turbulentFluxes::initialize");

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      const Qvar_ptr Q = m_Qvars[i];

     // skip if it already exists in the new_dw from a checkpoint
      if( new_dw->exists( Q->Qsum_Label, Q->matl, patch ) ){
        continue;
      }

      switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroAll<double>(  new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroAll<Vector>(  new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("ERROR:AnalysisModule:turbulentFluxes: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qvars
  }  // patches
}

//______________________________________________________________________
// This allows the user to restart from an uda in which this module was
// not enabled or add a variable to analyze.

void turbulentFluxes::scheduleRestartInitialize( SchedulerP    & sched,
                                                 const LevelP  & level)
{
  scheduleInitialize( sched, level);
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

  ProblemSpecP var_ps = m_ps->appendChild("variables");

  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];
    const string name = Q->Label->getName();
    ProblemSpecP newElem = var_ps->appendChild("variable");
    newElem->setAttribute( "name",             Q->Label->getName() );
    newElem->setAttribute( "firstSumTimestep", std::to_string( Q->firstSumTimestep ) );
    newElem->setAttribute( "nTimesteps",       std::to_string( Q->nTimesteps ) );
    newElem->setAttribute( "isStatEnabled",    std::to_string( Q->isStatEnabled ) );
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

  printSchedule(level,dbg_OTF_TF,"turbulentFluxes::sched_Q_mean");

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
    t->requires( Task::OldDW, Q->Qmean_Label,         matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->Q2mean_Label,        matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->Qu_Qv_Qw_mean_Label, matl, m_gn, 0 );

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
  //  Is it time?
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){
    carryForward_means( old_dw, new_dw, patches );
    return;
  }

  timeStep_vartype timeStep;
  old_dw->get(timeStep, m_timeStepLabel);
  const int curr_timestep = timeStep;

  //__________________________________
  //  Time to compute something
  for( auto p=0;p<patches->size();p++ ){

    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg_OTF_TF, "Doing turbulentFluxes::task_Q_mean");

    constCCVariable<Vector> vel;
    new_dw->get ( vel, m_velVar->Label, m_velVar->matl, patch, m_gn, 0 );

    for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
      Qvar_ptr Q = m_Qvars[i];

      set_firstSumTimestep( Q, curr_timestep );

      switch(Q->subtype->getType()) {

        case TypeDescription::double_type:{         // double
          Q_mean< double >( old_dw, new_dw, patch, Q, vel, curr_timestep );
          break;
        }
        case TypeDescription::Vector: {             // Vector
          Q_mean< Vector >( old_dw, new_dw, patch, Q, vel, curr_timestep );

          break;
        }
        default: {
          throw InternalError("ERROR:AnalysisModule:turbulentFluxes:: invalid data type", __FILE__, __LINE__);
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
                              constCCVariable<Vector> vel,
                              const int curr_timestep )

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

  int Q_ts = Q->firstSumTimestep;
  int timesteps = curr_timestep - Q_ts + 1;
  T nTimesteps(timesteps);

  Q->nTimesteps = timesteps;

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
    const int L = patch->getLevel()->getID();

    if ( c == m_monitorCell ){
      cout << "  turbulentFluxes:  " << m_monitorCell
           <<  " L-" << L << "  " << setw(10)
           << Q->Label->getName()
           << " nTimesteps: " << timesteps;

      cout.setf(ios::scientific,ios::floatfield);
      cout << setprecision( 10 )
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

  printSchedule(level,dbg_OTF_TF,"turbulentFluxes::sched_turbFluxes");

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

    t->requires( Task::OldDW, Q->variance_Label,    matl, m_gn, 0 );
    t->requires( Task::OldDW, Q->covariance_Label,  matl, m_gn, 0 );

    t->computes ( Q->variance_Label,   matl );
    t->computes ( Q->covariance_Label, matl );
  }
  sched->addTask( t, level->eachPatch() , m_matlSet );
}

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
    carryForward_variances( old_dw, new_dw, patches );
    return;
  }

  //__________________________________
  //
  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbg_OTF_TF, "Doing turbulentFluxes::task_turbFluxes");

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
          throw InternalError("ERROR:AnalysisModule:turbulentFluxes:: invalid data type", __FILE__, __LINE__);
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
    const int L = patch->getLevel()->getID();

    if ( c == m_monitorCell ){
      cout << "  turbulentFluxes:  " << m_monitorCell
           <<  " L-" << L << "  " << setw(10)
           << Q->Label->getName()
           <<"\t variance: "  << variance[c]
           <<"\t covariance: " << covariance[c] << endl;
    }
#endif
  }
}

//______________________________________________________________________
//                    UTILITIES
//______________________________________________________________________
// create the varlabel and populate the global name vector
VarLabel* turbulentFluxes::createVarLabel( const std::string name,
                                           const TypeDescription * td )
{
  VarLabel * label = VarLabel::create( name, td);
  m_VarLabelNames.push_back(name);

  return label;
}

//______________________________________________________________________
//  allocateAndZero all of the computed variables
template <class T>
void turbulentFluxes::allocateAndZeroAll( DataWarehouse* new_dw,
                                          const Patch  * patch,
                                          Qvar_ptr Q )
{
  int matl = Q->matl;
  if ( !Q->isInitialized ){
    allocateAndZero<T>(      new_dw, Q->Qsum_Label,          matl, patch );
    allocateAndZero<T>(      new_dw, Q->Q2sum_Label,         matl, patch );
    allocateAndZero<Vector>( new_dw, Q->Qu_Qv_Qw_sum_Label,  matl, patch );

    allocateAndZero<T>(      new_dw, Q->Qmean_Label,          matl, patch );
    allocateAndZero<T>(      new_dw, Q->Q2mean_Label,         matl, patch );
    allocateAndZero<Vector>( new_dw, Q->Qu_Qv_Qw_mean_Label,  matl, patch );

    allocateAndZero<T>(      new_dw, Q->variance_Label,       matl, patch );
    allocateAndZero<Vector>( new_dw, Q->covariance_Label,     matl, patch );
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
void turbulentFluxes::carryForward_means( DataWarehouse     * old_dw,
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
  }
}
//______________________________________________________________________
//  carryForward  variables from old_dw to new_dw
void turbulentFluxes::carryForward_variances( DataWarehouse     * old_dw,
                                              DataWarehouse     * new_dw,
                                              const PatchSubset * patches )
{
  for ( unsigned int i =0 ; i < m_Qvars.size(); i++ ) {
    const Qvar_ptr Q = m_Qvars[i];

    new_dw->transferFrom(old_dw, Q->variance_Label,       patches, Q->matlSubset );
    new_dw->transferFrom(old_dw, Q->covariance_Label,     patches, Q->matlSubset );
  }
}

//______________________________________________________________________
//
void turbulentFluxes::set_firstSumTimestep(Qvar_ptr Q,
                                           const int curr_timestep)
{

  if( Q->isStatEnabled ){
    return;
  }

  //__________________________________
  //
  static int count = 0;

  proc0cout_cmp(count, 0) << "________________________DataAnalysis: TurbulentFluxes\n"
                          << " Started computing the variance & covariance for:\n";
  Q->isStatEnabled = true;
  Q->firstSumTimestep = curr_timestep;
  Q->print();

  count ++;
  proc0cout_cmp(count, (int) m_Qvars.size()) << "________________________DataAnalysis: TurbulentFluxes\n";


}
