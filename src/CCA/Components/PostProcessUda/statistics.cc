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

#include <CCA/Components/PostProcessUda/statistics.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
/*______________________________________________________________________
  The post processing module will compute the mean, variance, skewness and Kurtosis
  for a set of CCVariables in an existing uda over the timesteps in the uda.
  The usage is:

   sus -postProcessUda <uda>

   Make the following changes to the <uda>/input.xml

  <SimulationComponent type="postProcessUda"/>

  <save label="mean_press_CC"/>
  <save label="variance_press_CC"/>
  <save label="skewness_press_CC"/>
  <save label="kurtosis_press_CC"/>

  <save label="mean_vel_CC"/>
  <save label="variance_vel_CC"/>
  <save label="skewness_vel_CC"/>
  <save label="kurtosis_vel_CC"/>

  <PostProcess>
    <Module type = "statistics">
      <timeStart>   2e-9    </timeStart>
      <timeStop>   100      </timeStop>
      <material>    Air     </material>              << A)
      <materialIndex> 0     </materialIndex>         << B)   You must specifie either A or B
      <monitorCell> [0,0,0] </monitorCell>           << Used to monitor calculations in one cell
      <computeHigherOrderStats> true </computeHigherOrderStats>   << Needed to compute skewness and Kurtosis
      <Variables>
        <analyze label="press_CC"  matl="0"/>        << Variables of interest
        <analyze label="vel_CC"    matl="0"/>
      </Variables>
    </Module>
  </PostProcess>

______________________________________________________________________*/

using namespace Uintah;
using namespace postProcess;
using namespace std;

static DebugStream dbg("POSTPROCESS_STATISTICS", false);
//______________________________________________________________________
statistics::statistics(ProblemSpecP    & module_spec,
                       MaterialManagerP& materialManager,
                       Output          * dataArchiver,
                       DataArchive     * dataArchive)
  : Module(module_spec, materialManager, dataArchiver, dataArchive)
{
  d_prob_spec = module_spec;

  // Time Step
  m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription());

  // Simulation Time
  m_simulationTimeLabel = VarLabel::create(simTime_name, simTime_vartype::getTypeDescription());
}

//______________________________________________________________________
//
statistics::~statistics()
{
  dbg << " Doing: destorying statistics " << endl;
  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }

  // delete each Qstats label
  for (unsigned int i =0 ; i < d_Qstats.size(); i++) {
    Qstats& Q = d_Qstats[i];
    VarLabel::destroy( Q.Qsum_Label );
    VarLabel::destroy( Q.Qsum2_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qvariance_Label );

    if( d_doHigherOrderStats ){
      VarLabel::destroy( Q.Qsum3_Label );
      VarLabel::destroy( Q.Qsum4_Label );
    }
  }

  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void statistics::problemSetup()
{
  dbg << "Doing problemSetup \t\t\t\tstatistics" << endl;

  proc0cout << "__________________________________ Post Process module: statistics" << endl;
  readTimeStartStop(d_prob_spec, d_startTime, d_stopTime);

  d_matlSet = scinew MaterialSet();
  map<string,int> Qmatls;

  createMatlSet( d_prob_spec, d_matlSet, Qmatls );
  proc0cout << "  StartTime: " << d_startTime << " stopTime: "<< d_stopTime << " " << *d_matlSet << endl;

  // debugging
  d_prob_spec->get("monitorCell", d_monitorCell);

  d_prob_spec->get("computeHigherOrderStats", d_doHigherOrderStats );
  if (d_doHigherOrderStats){
    proc0cout << "  Computing 2nd, 3rd and 4th order statistics for all of the variables listed"<< endl;
  } else {
    proc0cout << "  Computing 2nd order statistics for all of the variables listed"<< endl;
  }

  //__________________________________
  //  Read in variables label names
   ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("statistics: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {
    map<string,string> attribute;
    var_spec->getAttributes(attribute);

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
      warn << "ERROR:Module:statisticst: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    Qstats Q;
    Q.matl    = Qmatls[name];
    Q.Q_Label = label;
    Q.subtype = subtype;
    Q.initializeTimestep();          // initialize the start timestep = 0;

    Q.Qsum_Label      = VarLabel::create( "sum_" + name,      td);
    Q.Qsum2_Label     = VarLabel::create( "sum2_" + name,     td);
    Q.Qmean_Label     = VarLabel::create( "mean_" + name,     td);
    Q.Qvariance_Label = VarLabel::create( "variance_" + name, td);

    if( d_doHigherOrderStats ){
      Q.Qsum3_Label     = VarLabel::create( "sum3_" + name,      td);
      Q.Qskewness_Label = VarLabel::create( "skewness_" + name,  td);

      Q.Qsum4_Label     = VarLabel::create( "sum4_" + name,      td);
      Q.Qkurtosis_Label = VarLabel::create( "kurtosis_" + name,  td);
    }
    d_Qstats.push_back( Q );

    //__________________________________
    //  bulletproofing
    std::string variance = "variance_"+ name;
    std::string skew     = "skewness_"+ name;
    std::string kurtosis = "kurtosis_"+ name;
    ostringstream mesg;
    mesg << "";
    if( !d_dataArchiver->isLabelSaved( variance ) ){
      mesg << variance;
    }
    if( !d_dataArchiver->isLabelSaved( skew )  && d_doHigherOrderStats){
      mesg << " " << skew;
    }
    if( !d_dataArchiver->isLabelSaved( kurtosis ) && d_doHigherOrderStats){
      mesg << " " << kurtosis;
    }

    if( mesg.str() != "" ){
      ostringstream warn;
      warn << "WARNING:  You've activated the DataAnalysis:statistics module but your not saving the variable(s) ("
           << mesg.str() << ")";
      proc0cout << warn.str() << endl;
    }
  }
  proc0cout << "__________________________________ Post Process module: statistics" << endl;
}

//______________________________________________________________________
//
void statistics::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level)
{
  printSchedule( level,dbg,"statistics::scheduleInitialize" );

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
    printTask(patches, patch,dbg,"Doing statistics::initialize");

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
  }  // pathes
}

//______________________________________________________________________
//
void statistics::scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level)
{
  printSchedule( level,dbg,"statistics::scheduleDoAnalysis" );

  Task* t = scinew Task("statistics::doAnalysis",
                   this,&statistics::doAnalysis);

  t->requires( Task::OldDW, m_timeStepLabel);
  t->requires( Task::OldDW, m_simulationTimeLabel);

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

    t->computes ( Q.Qsum_Label,       matSubSet );
    t->computes ( Q.Qsum2_Label,      matSubSet );
    t->computes ( Q.Qmean_Label,      matSubSet );
    t->computes ( Q.Qvariance_Label,  matSubSet );

    //__________________________________
    // Higher order statistics
    if( d_doHigherOrderStats ){

      t->requires( Task::OldDW, Q.Qsum3_Label, matSubSet, gn, 0 );
      t->requires( Task::OldDW, Q.Qsum4_Label, matSubSet, gn, 0 );

      t->computes ( Q.Qsum3_Label,     matSubSet );
      t->computes ( Q.Qsum4_Label,     matSubSet );
      t->computes ( Q.Qskewness_Label, matSubSet );
      t->computes ( Q.Qkurtosis_Label, matSubSet );
    }
    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
//    Q.print();
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

    printTask(patches, patch,dbg,"Doing statistics::doAnalysis");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          computeStatsWrapper< double >(old_dw, new_dw, patches, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          computeStatsWrapper< Vector >(old_dw, new_dw, patches,  patch, Q);
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
  simTime_vartype simTime;
  old_dw->get( simTime, m_simulationTimeLabel );

  if(simTime < d_startTime || simTime > d_stopTime){
    //proc0cout << " IGNORING------------DataAnalysis: Statistics" << endl;
    allocateAndZeroStats<T>( new_dw, patch, Q);
    allocateAndZeroSums<T>(  new_dw, patch, Q);
  }else {
    //proc0cout << " Computing------------DataAnalysis: Statistics" << endl;

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
  timeStep_vartype timeStep;
  old_dw->get( timeStep, m_timeStepLabel );

  static proc0patch0cout mesg( d_Qstats.size() );
  ostringstream msg;
  msg <<"    statistics::computeStats( "<< Q.Q_Label->getName() << " )\n";
  mesg.print(patch, msg );
  //__________________________________
  
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
  CCVariable< T > Qvariance;

  new_dw->allocateAndPut( Qsum,      Q.Qsum_Label,      matl, patch );
  new_dw->allocateAndPut( Qsum2,     Q.Qsum2_Label,     matl, patch );
  new_dw->allocateAndPut( Qmean,     Q.Qmean_Label,     matl, patch );
  new_dw->allocateAndPut( Qvariance, Q.Qvariance_Label, matl, patch );

  Q.setStart(timeStep);
  int Q_ts = Q.getStart();
  int ts = timeStep - Q_ts + 1;

  T nTimesteps(ts);

  //__________________________________
  //  Lower order stats  1st and 2nd
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    T me = Qvar[c];     // for readability
    Qsum[c]    = me + Qsum_old[c];
    Qmean[c]   = Qsum[c]/nTimesteps;

    Qsum2[c]   = me * me + Qsum2_old[c];
    T Qmean2   = Qsum2[c]/nTimesteps;

    Qvariance[c] = Qmean2 - Qmean[c] * Qmean[c];
  }

  //__________________________________
  //  debugging
  if ( d_monitorCell != IntVector(-9,-9,-9) && patch->containsCell (d_monitorCell) ){
    IntVector c = d_monitorCell;
    cout << "  stats:  " << c <<  setw(10) << Q.Q_Label->getName()
         << " time step: " << ts
         <<"\t time step " <<  timeStep
         << " d_startTime: " << d_startTime << "\n"
         <<"\t Q_var: " << Qvar[c]
         <<"\t Qsum: "  << Qsum[c]
         <<"\t Qsum2: "  << Qsum2[c]
         <<"\t Qmean: " << Qmean[c]
         <<"\t Qvariance: " << Qvariance[c]
         << endl;
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

    CCVariable< T > Qskewness;
    CCVariable< T > Qkurtosis;
    new_dw->allocateAndPut( Qsum3,     Q.Qsum3_Label,     matl, patch );
    new_dw->allocateAndPut( Qsum4,     Q.Qsum4_Label,     matl, patch );
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
      T Qmean3  = Qsum3[c]/nTimesteps;

      Qskewness[c] = Qmean3 - Qbar3 - 3 * Qvariance[c] * Qbar;

      // kurtosis
      Qsum4[c]  = me2 * me2 + Qsum4_old[c];
      T Qmean4  = Qsum4[c]/nTimesteps;

      Qkurtosis[c] = Qmean4 - Qbar4
                   - 6 * Qvariance[c] * Qbar2
                   - 4 * Qskewness[c] * Qbar;
    }

    //__________________________________
    //  debugging
    if ( d_monitorCell != IntVector(-9,-9,-9) && patch->containsCell (d_monitorCell)){
      IntVector c = d_monitorCell;
      cout <<"\t Qsum3: "  << Qsum3[c]
           <<"\t Qsum4: "  << Qsum4[c]
           <<"\t Qskewness: " << Qskewness[c]
           <<"\t Qkurtosis: " << Qkurtosis[c]
           << endl;
    }
  }
}

//______________________________________________________________________
//  allocateAndZero  statistics variables
template <class T>
void statistics::allocateAndZeroStats(DataWarehouse * new_dw,
                                      const Patch   * patch,
                                      const Qstats  & Q )
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
void statistics::allocateAndZeroSums( DataWarehouse * new_dw,
                                      const Patch   * patch,
                                      const Qstats  & Q )
{
  int matl = Q.matl;
  allocateAndZero<T>( new_dw, Q.Qsum_Label,  matl, patch );
  allocateAndZero<T>( new_dw, Q.Qsum2_Label, matl, patch );
//    proc0cout << "    Statistics: " << Q.Q_Label->getName() << " initializing low order sums on patch: " << patch->getID()<<endl;


  if( d_doHigherOrderStats ){
    allocateAndZero<T>( new_dw, Q.Qsum3_Label, matl, patch );
    allocateAndZero<T>( new_dw, Q.Qsum4_Label, matl, patch );
//    proc0cout << "    Statistics: " << Q.Q_Label->getName() << " initializing high order sums on patch: " << patch->getID() << endl;
  }
}
