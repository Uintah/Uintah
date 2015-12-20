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
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <cstdio>


using namespace Uintah;
using namespace std;
static DebugStream cout_doing("STATISTICS_DOING_COUT", false);
static DebugStream cout_dbg("STATISTICS_DBG_COUT", false);
//______________________________________________________________________
statistics::statistics(ProblemSpecP& module_spec,
                       SimulationStateP& sharedState,
                       Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matlSet     = 0;
  d_startTime   = 0;
  d_stopTime    = DBL_MAX;
  d_doHigherOrderStats = false;
  d_startTimeTimestep = 0;
  
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
    Qstats Q = d_Qstats[i];
    VarLabel::destroy( Q.Qsum_Label );
    VarLabel::destroy( Q.Qsum2_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qvariance_Label );

    if( d_doHigherOrderStats ){
      VarLabel::destroy( Q.Qsum3_Label );
      VarLabel::destroy( Q.Qmean3_Label );
      VarLabel::destroy( Q.Qsum4_Label );
      VarLabel::destroy( Q.Qmean4_Label );
    }

    if ( Q.isVelocityLabel ){
      VarLabel::destroy( d_uv_primeLabel );
      VarLabel::destroy( d_uw_primeLabel );
      VarLabel::destroy( d_vw_primeLabel );
    }
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void statistics::problemSetup(const ProblemSpecP& prob_spec,
                              const ProblemSpecP& ,
                              GridP& grid,
                              SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tstatistics" << endl;

  int numMatls  = d_sharedState->getNumMatls();
  if(!d_dataArchiver){
    throw InternalError("statistics:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("timeStart",  d_startTime);
  d_prob_spec->require("timeStop",   d_stopTime);
  
  // a backdoor to set when the module was initiated.
  // this is a quick fix until I find a better way.
  d_prob_spec->getWithDefault("startTimeTimestep",d_startTimeTimestep, 0);

  // Start time < stop time
  if(d_startTime > d_stopTime ){
    throw ProblemSetupException("\n ERROR:statistics: startTime > stopTime. \n", __FILE__, __LINE__);
  }

  //__________________________________
  // find the material to extract data from.  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>

  Material* matl = NULL;

  if(d_prob_spec->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }

  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  proc0cout << "__________________________________ Data Analysis module: statistics" << endl;
  d_prob_spec->get("computeHigherOrderStats", d_doHigherOrderStats );
  if (d_doHigherOrderStats){

    proc0cout << "         Computing 2nd, 3rd and 4th order statistics for all of the variables listed"<< endl;
  } else {
    proc0cout << "         Computing 2nd order statistics for all of the variables listed"<< endl;
  }

  //__________________________________
  //  Read in variables label names

   ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("statistics: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0;
                    var_spec = var_spec->findNextBlock("analyze")) {
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
    if(label == NULL){
      throw ProblemSetupException("statistics label not found: " + name , __FILE__, __LINE__);
    }

    //__________________________________
    // Only CCVariable Doubles and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType() != TypeDescription::CCVariable  ||
       ( subtype->getType() != TypeDescription::double_type &&
         subtype->getType() != TypeDescription::Vector)  ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:statisticst: ("<<label->getName() << " "
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    Qstats Q;
    Q.name    = name;
    Q.matl    = matl;
    Q.Q_Label = label;
    Q.subtype = subtype;
    Q.isVelocityLabel = false;

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
      Q.isVelocityLabel    = true;
      d_uv_primeLabel = VarLabel::create( "uv_prime",  CCVariable<double>::getTypeDescription());
      d_uw_primeLabel = VarLabel::create( "uw_prime",  CCVariable<double>::getTypeDescription());
      d_vw_primeLabel = VarLabel::create( "vw_prime",  CCVariable<double>::getTypeDescription());
      proc0cout << "         Computing uv_prime, uw_prime, vw_prime using ("<< name << ")" << endl;
    }

    //__________________________________
    // keep track of which summation variables
    // have been initialized.  A user can
    // add a variable on a restart.  Default is false.
    Q.isInitialized[lowOrder]  = false;
    Q.isInitialized[highOrder] = false;
    
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
      warn << "\nWARNING:  You've activated the DataAnalysis:statistics module but your not saving the variable(s) ("
           << mesg.str() << ")\n";
      proc0cout << warn.str() << endl;
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
}

//______________________________________________________________________
void statistics::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleInitialize" );

  Task* t = scinew Task("statistics::initialize",
                   this,&statistics::initialize);

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];

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
    printTask(patches, patch,cout_doing,"Doing statistics::initialize");

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
  const Uintah::PatchSet* const ps = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
  int rank = Parallel::getMPIRank();
  const PatchSubset* myPatches = ps->getSubset(rank);
  const Patch* firstPatch = myPatches->get(0);
  

  Task* t = scinew Task("statistics::restartInitialize",
                   this,&statistics::restartInitialize);

  bool addTask = false;
  
  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];
    
    // does sumVariables exist in checkpoint
    //              low order
    if (new_dw->exists( Q.Qsum_Label, Q.matl, firstPatch) ){
      Q.isInitialized[lowOrder] = true;
    }
    
    
    //              high order
    if( new_dw->exists( Q.Qsum3_Label, Q.matl, firstPatch) ){
      Q.isInitialized[highOrder] = true;
    }
    
   
    // if the Q.sum was not in previous checkpoint
    if( !Q.isInitialized[lowOrder] ){
      t->computes ( Q.Qsum_Label );
      t->computes ( Q.Qsum2_Label );
      addTask = true;
      cout << "    Adding lowOrder computes for " << Q.name << "  " << d_Qstats[i].isInitialized[lowOrder] << endl;
    }

    if( d_doHigherOrderStats && !Q.isInitialized[highOrder] ){
      t->computes ( Q.Qsum3_Label );
      t->computes ( Q.Qsum4_Label );                                   
      addTask = true;                                                  
      cout << "    Adding highOrder computes for " << Q.name << "  " << d_Qstats[i].isInitialized[highOrder] << endl;  
    }
  }
  
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
#if 1
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
  }  // pathes
#endif
}


void statistics::restartInitialize()
{
}

//______________________________________________________________________
void statistics::scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleDoAnalysis" );

  Task* t = scinew Task("statistics::doAnalysis",
                   this,&statistics::doAnalysis);

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
    t->computes ( Q.Qmean2_Label,     matSubSet );
    t->computes ( Q.Qvariance_Label,  matSubSet );

    //__________________________________
    // Higher order statistics
    if( d_doHigherOrderStats ){

      t->requires( Task::OldDW, Q.Qsum3_Label, matSubSet, gn, 0 );
      t->requires( Task::OldDW, Q.Qsum4_Label, matSubSet, gn, 0 );

      t->computes ( Q.Qsum3_Label,     matSubSet );
      t->computes ( Q.Qsum4_Label,     matSubSet );
      t->computes ( Q.Qmean3_Label,    matSubSet );
      t->computes ( Q.Qmean4_Label,    matSubSet );
      t->computes ( Q.Qskewness_Label, matSubSet );
      t->computes ( Q.Qkurtosis_Label, matSubSet );
    }

    //__________________________________
    //  Reynolds Stress Terms
    if ( Q.isVelocityLabel ){
      t->computes ( d_uv_primeLabel, matSubSet );
      t->computes ( d_uw_primeLabel, matSubSet );
      t->computes ( d_vw_primeLabel, matSubSet );
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

    printTask(patches, patch,cout_doing,"Doing statistics::doAnalysis");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          computeStatsWrapper< double >(old_dw, new_dw, patches, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          computeStatsWrapper< Vector >(old_dw, new_dw, patches,  patch, Q);

          computeReynoldsStress( new_dw,patch, Q);

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
                                      Qstats Q)
{
  double now = d_dataArchiver->getCurrentTime();

  if(now < d_startTime || now > d_stopTime){

//    proc0cout << " IGNORING------------DataAnalysis: Statistics" << endl;
    allocateAndZeroStats<T>( new_dw, patch, Q);
    carryForwardSums( old_dw, new_dw, patches, Q );

  }else {
//    proc0cout << " Computing------------DataAnalysis: Statistics" << endl;

    if ( d_startTimeTimestep == 0 ){
      d_startTimeTimestep = d_sharedState->getCurrentTopLevelTimeStep();
    }

    computeStats< T >(old_dw, new_dw, patch, Q);
  }
}

//______________________________________________________________________
//
template <class T>
void statistics::computeStats( DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const Patch*    patch,
                               Qstats Q)
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
  int timestep = d_sharedState->getCurrentTopLevelTimeStep() - d_startTimeTimestep + 1;

  T nTimesteps(timestep);

  //__________________________________
  //  Lower order stats  1st and 2nd
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    T me = Qvar[c];     // for readability
    Qsum[c]    = me + Qsum_old[c];
    Qmean[c]   = Qsum[c]/nTimesteps;

    Qsum2[c]  = me * me + Qsum2_old[c];
    Qmean2[c] = Qsum2[c]/nTimesteps;

    Qvariance[c] = Qmean2[c] - Qmean[c] * Qmean[c];

  
#if 0
    //__________________________________
    //  debugging    
    if ( c == IntVector ( -1, 75, 0) ){
      cout << Q.name << " nTimestep: " << nTimesteps 
           <<  " topLevelTimestep " <<  d_sharedState->getCurrentTopLevelTimeStep()
           << " d_startTimestep: " << d_startTimeTimestep
           <<" Q_var: " << me << " Qsum: " << Qsum[c]<< " Qmean: " << Qmean[c] << endl;
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
      T Qbar3 = Qbar * Qbar * Qbar;
      T Qbar4 = Qbar * Qbar * Qbar * Qbar;

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
//  Computes u'v', u'w', v'w'
void statistics::computeReynoldsStress( DataWarehouse* new_dw,
                                        const Patch* patch,
                                        Qstats Q)
{
  if ( Q.isVelocityLabel == false ){
    return;
  }

  const int matl = Q.matl;

  constCCVariable<Vector> vel_CC;

  Ghost::GhostType  gn  = Ghost::None;
  new_dw->get ( vel_CC, Q.Q_Label, matl, patch, gn, 0 );

  CCVariable< double > uv_prime;
  CCVariable< double > uw_prime;
  CCVariable< double > vw_prime;

  new_dw->allocateAndPut( uv_prime, d_uv_primeLabel,  matl, patch );
  new_dw->allocateAndPut( uw_prime, d_uw_primeLabel,  matl, patch );
  new_dw->allocateAndPut( vw_prime, d_vw_primeLabel,  matl, patch );

  // is it time to compute?
  double oneZero = 1.0;

  double now = d_dataArchiver->getCurrentTime();
  if(now < d_startTime || now > d_stopTime){
    oneZero = 0.0;
  }

  // do it
  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    Vector vel = vel_CC[c];  // for readability
    uv_prime[c] = vel.x() * vel.y() * oneZero;
    uw_prime[c] = vel.x() * vel.z() * oneZero;
    vw_prime[c] = vel.y() * vel.z() * oneZero;
  }
}

//______________________________________________________________________
//  allocateAndZero  statistics variables
template <class T>
void statistics::allocateAndZeroStats( DataWarehouse* new_dw,
                                      const Patch* patch,
                                      Qstats Q )
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
                                      Qstats Q )
{
  int matl = Q.matl;
  if ( !Q.isInitialized[lowOrder] ){
    allocateAndZero<T>( new_dw, Q.Qsum_Label,  matl, patch );
    allocateAndZero<T>( new_dw, Q.Qsum2_Label, matl, patch );
    cout << "__________________________________ " << Q.name << " initializing low order sums Patch: " << patch->getID()<< endl;
  }
  
  if( d_doHigherOrderStats && !Q.isInitialized[highOrder] ){
    allocateAndZero<T>( new_dw, Q.Qsum3_Label, matl, patch );
    allocateAndZero<T>( new_dw, Q.Qsum4_Label, matl, patch );
    cout << "__________________________________" << Q.name << " initializing high order sums Patch: " << patch->getID() << endl;
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
                                   Qstats Q )
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
