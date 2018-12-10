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

#include <CCA/Components/PostProcessUda/spatioTemporalAvg.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/StringUtil.h>

#include <iostream>
#include <iomanip>
/*______________________________________________________________________
  The post processing module will compute the mean, variance, over a region of space
  for a set of CCVariables in an existing uda over the timesteps in the uda.
  The usage is:

   sus -postProcessUda <uda>

   Make the following changes to the <uda>/input.xml

  <SimulationComponent type="postProcessUda"/>

  <save label="avg_press_CC"/>
  <save label="avg_variance_press_CC"/>

  <save label="avg_vel_CC"/>
  <save label="avg_variance_vel_CC"/>

  <PostProcess>
    <Module type = "spatioTemporalAvg">
      <timeStart>   2e-9     </timeStart>
      <timeStop>   100       </timeStop>
      <material>    Air      </material>
      <domain>   everywhere  </domain>        << options: everwhere, interior, boundaries
      <avgBoxCells>  [5,5,5] </avgBoxCells>   << size of box to average over, cells.

      <Variables>
        <analyze label="press_CC"  matl="0"/>
        <analyze label="vel_CC"    matl="0"/>
      </Variables>
    </Module>
  </PostProcess>

______________________________________________________________________*/

using namespace Uintah;
using namespace postProcess;
using namespace std;

static DebugStream dbg("POSTPROCESS_SPATIOTEMPORALAVG", false);
//______________________________________________________________________
spatioTemporalAvg::spatioTemporalAvg(ProblemSpecP    & module_spec,
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
spatioTemporalAvg::~spatioTemporalAvg()
{
  dbg << " Doing: destorying spatioTemporalAvg " << endl;
  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }

  // delete each Qstats label
  for (unsigned int i =0 ; i < d_Qstats.size(); i++) {
    Qstats& Q = d_Qstats[i];
    VarLabel::destroy( Q.avgLabel );
    VarLabel::destroy( Q.varianceLabel );
  }

  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void spatioTemporalAvg::problemSetup()
{
  dbg << "Doing problemSetup \t\t\t\tspatioTemporalAvg" << endl;

  proc0cout << "__________________________________ Post Process module: spatioTemporalAvg" << endl;
  proc0cout << "         Computing spatial Temporal average for all of the variables listed"<< endl;

  //__________________________________
  //  region to average over
  d_prob_spec->require("avgBoxCells", d_avgBoxCells );

  //__________________________________
  //  domain
  string ans = "null";
  d_prob_spec->get( "domain", ans );

  ans = string_toupper(ans);
  if ( ans == "EVERYWHERE" ){
    d_compDomain = EVERYWHERE;
    proc0cout << "  Computational Domain: Everywhere\n";
  }
  else if ( ans == "INTERIOR" ){
    d_compDomain = INTERIOR;
    proc0cout << "  Computational Domain: InteriorCells\n";
  }
  else if ( ans == "BOUNDARIES" ) {
    d_compDomain = BOUNDARIES;
    proc0cout << "  Computational Domain: Domain boundaries\n";
  }

  //__________________________________
  //  Parse ups for time variables and matlSet
  readTimeStartStop(d_prob_spec, d_startTime, d_stopTime);


  d_matlSet = scinew MaterialSet();
  map<string,int> Qmatls;

  createMatlSet( d_prob_spec, d_matlSet, Qmatls );
  proc0cout << "  StartTime: " << d_startTime << " stopTime: "<< d_stopTime << " " << *d_matlSet << endl;

  //__________________________________
  //  
  ProblemSpecP temp_ps = d_prob_spec->findBlock("TemporalAvg");
  map<string,string> attribute;
  temp_ps->getAttributes(attribute);
  if ( attribute["OnOff"] == "on" ){
    d_doTemporalAvg = true;
    temp_ps->require( "baseTimestep", d_baseTimestep );
    proc0cout << "  Temporal Average:  baseTimestep: " << d_baseTimestep << endl;
  }
  

  // debugging
  d_prob_spec->get("monitorCell", d_monitorCell);

  //__________________________________
  //  Read in variables label names
   ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("spatioTemporalAvg: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {
    map<string,string> attribute;
    var_spec->getAttributes(attribute);

    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == nullptr){
      throw ProblemSetupException("spatioTemporalAvg label not found: " + name , __FILE__, __LINE__);
    }

    //__________________________________
    // Only CCVariable doubles, floats and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType() != TypeDescription::CCVariable  ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::float_type &&
          subtype->getType() != TypeDescription::Vector ) ) {
      ostringstream warn;
      warn << "ERROR:Module:spatioTemporalAvg: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    Qstats Q;
    Q.matl    = Qmatls[name];
    Q.Q_Label = label;
    Q.subtype = subtype;
    Q.initializeTimestep();          // initialize the start timestep = 0;

    Q.avgLabel      = VarLabel::create( "avg_" + name,     td);
    Q.varianceLabel = VarLabel::create( "avg_variance_" + name, td);
    d_Qstats.push_back( Q );

    //__________________________________
    //  bulletproofing
    std::string variance = "spatioTemporalAvg_variance_"+ name;
    ostringstream mesg;
    mesg << "";
    if( !d_dataArchiver->isLabelSaved( variance ) ){
      mesg << variance;
    }

    if( mesg.str() != "" ){
      ostringstream warn;
      warn << "WARNING:  You've activated the postProcess:spatioTemporalAvg module but your not saving the variable(s) ("
           << mesg.str() << ")";
      proc0cout << warn.str() << endl;
    }
  }
  proc0cout << "__________________________________ Post Process module: spatioTemporalAvg" << endl;

}

//______________________________________________________________________
void spatioTemporalAvg::scheduleInitialize(SchedulerP   & sched,
                                           const LevelP & level)
{
  printSchedule( level,dbg,"spatioTemporalAvg::scheduleInitialize" );

  d_lb = sched->getLoadBalancer();

  Task* t = scinew Task("spatioTemporalAvg::initialize",
                   this,&spatioTemporalAvg::initialize);



  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    const Qstats Q = d_Qstats[i];
    t->computes ( Q.avgLabel );
    t->computes ( Q.varianceLabel );
  }
  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
//
void spatioTemporalAvg::initialize(const ProcessorGroup  *,
                                   const PatchSubset     * patches,
                                   const MaterialSubset  *,
                                   DataWarehouse         *,
                                   DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dbg,"Doing spatioTemporalAvg::initialize");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZeroLabels<double>( new_dw,  patch,  Q );
          break;
        }
        case TypeDescription::float_type:{         // float
          allocateAndZeroLabels<float>( new_dw,  patch,  Q );
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZeroLabels<Vector>( new_dw,  patch,  Q );
          break;
        }
        default: {
          throw InternalError("spatioTemporalAvg: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstat
  }  // pathes
}

//______________________________________________________________________
//
void spatioTemporalAvg::scheduleDoAnalysis(SchedulerP   & sched,
                                           const LevelP & level)
{
  printSchedule( level,dbg,"spatioTemporalAvg::scheduleDoAnalysis" );

  Task* t = scinew Task("spatioTemporalAvg::doAnalysis",
                   this,&spatioTemporalAvg::doAnalysis);

  t->requires( Task::OldDW, m_timeStepLabel);
  t->requires( Task::OldDW, m_simulationTimeLabel);

  Ghost::GhostType  gn  = Ghost::None;

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];

    // define the matl subset for this variable
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( Q.matl );
    matSubSet->addReference();

    t->requires( Task::NewDW, Q.Q_Label, matSubSet, gn, 0 );
    t->computes ( Q.avgLabel,        matSubSet );
    t->computes ( Q.varianceLabel,  matSubSet );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
//    Q.print();
  }
  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
// Compute the spatioTemporalAvg for each variable the user requested
void spatioTemporalAvg::doAnalysis(const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * ,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dbg,"Doing spatioTemporalAvg::doAnalysis");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats& Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          computeAvgWrapper<double>(old_dw, new_dw, patches, patch, Q);
          break;
        }
        case TypeDescription::float_type:{         // float
          computeAvgWrapper<float>(old_dw, new_dw, patches, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          computeAvgWrapper<Vector>(old_dw, new_dw, patches,  patch, Q);
          break;
        }
        default: {
          throw InternalError("spatioTemporalAvg: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // qstats loop
  }  // patches
}

//______________________________________________________________________
//  computeAvgWrapper:
template <class T>
void spatioTemporalAvg::computeAvgWrapper( DataWarehouse     * old_dw,
                                           DataWarehouse     * new_dw,
                                           const PatchSubset * patches,
                                           const Patch       * patch,
                                           Qstats& Q)
{
  simTime_vartype simTime;
  old_dw->get( simTime, m_simulationTimeLabel );
  
  if(simTime < d_startTime || simTime > d_stopTime){
    //proc0cout << " IGNORING------------DataAnalysis: spatioTemporalAvg" << endl;
    allocateAndZeroLabels< T >( new_dw, patch, Q );
  }else {
    //proc0cout << " Computing------------DataAnalysis: spatioTemporalAvg" << endl;

    computeAvg< T >(old_dw, new_dw, patch, Q);
  }
}



//______________________________________________________________________
//
template <class T>
void spatioTemporalAvg::computeAvg( DataWarehouse  * old_dw,
                                    DataWarehouse  * new_dw,
                                    const Patch    * patch,
                                    Qstats& Q)
{
  timeStep_vartype timeStep;
  old_dw->get( timeStep, m_timeStepLabel );

  static proc0patch0cout mesg( d_Qstats.size() );
  ostringstream msg;
  msg <<"    spatioTemporalAvg::computeAvg( "<< Q.Q_Label->getName() << " )\n";
  mesg.print(patch, msg );
  //__________________________________
  
  
  const int matl = Q.matl;

  constCCVariable<T> Qvar_old;
 
  if ( d_doTemporalAvg ){
  
//    cout << "   OLD_DW: " << old_dw->getID() <<  " exists: " << old_dw->exists( Q.Q_Label, matl, patch) << endl;
    old_dw->get ( Qvar_old, Q.Q_Label, matl, patch, Ghost::None, 0 );
  }
  
  bool doComputeTimeAverage = ( d_doTemporalAvg && timeStep >= (unsigned int) d_baseTimestep );
  
  constCCVariable<T> Qvar;
  new_dw->get ( Qvar,     Q.Q_Label, matl, patch, Ghost::None, 0 );
  
  CCVariable< T > Qavg;
  CCVariable< T > Qvariance;
  
  new_dw->allocateAndPut( Qavg,      Q.avgLabel,     matl, patch );
  new_dw->allocateAndPut( Qvariance, Q.varianceLabel, matl, patch );

  T zero = T(0.0);
  Qavg.initialize( zero );
  Qvariance.initialize( zero );

  //__________________________________
  //  compute the number boxes to avg over
  if (d_compDomain == EVERYWHERE || d_compDomain == INTERIOR ){

    CellIterator iter = patch->getCellIterator();

    if( doComputeTimeAverage )
      computeTimeAverage<T>( patch, iter, Qvar, Qvar_old, Qavg, timeStep);
    spatioTemporalAvg::query( patch, Qvar, Qavg, Qvariance, d_avgBoxCells, iter);
  }

  //__________________________________
  //  Loop over the boundary faces that this patch owns
  if (d_compDomain == EVERYWHERE || d_compDomain == BOUNDARIES ){
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
      Patch::FaceType face = *itr;

      IntVector avgPanelCells = d_avgBoxCells;
      IntVector axes = patch->getFaceAxes(face);
      int pDir = axes[0];
      avgPanelCells[pDir] = 1;

      CellIterator iter = patch->getFaceIterator( face, Patch::ExtraMinusEdgeCells );
      if( doComputeTimeAverage )
        computeTimeAverage<T>( patch, iter, Qvar, Qvar_old, Qavg, timeStep);
      spatioTemporalAvg::query( patch, Qvar, Qavg, Qvariance, avgPanelCells, iter);

    }  // face loop
  }

  //__________________________________
  //  debugging
  if ( d_monitorCell != IntVector(-9,-9,-9) && patch->containsCell (d_monitorCell) ){
    IntVector c = d_monitorCell;
    cout << "  stats:  " << c <<  setw(10)<< Q.Q_Label->getName()
         <<"\t time step " <<  timeStep
         << " d_startTime: " << d_startTime << "\n"
         <<"\t Q_var: " << Qvar[c]
         <<"\t Qavg: "  << Qavg[c]
//         <<"\t Qvariance: " << Qvariance[c]
         << endl;
  }
}


//______________________________________________________________________
//
template <class T>
void spatioTemporalAvg::computeTimeAverage( const Patch         * patch,
                                            CellIterator        & iter,
                                            constCCVariable< T >& Qvar,
                                            constCCVariable< T >& Qvar_old,
                                            CCVariable< T >     & Qavg,
                                            const int           & timeStep )
{

  T deltaTime = T( d_udaTimes[timeStep] - d_udaTimes[d_baseTimestep] );
 
#if 0
  proc0cout << " timeStep: " << timeStep << " udaTime: " << d_udaTimes[timeStep]
            << " d_baseTimestep: " << d_baseTimestep << " baseTime: " << d_udaTimes[d_baseTimestep]
            << " deltaTime: " << deltaTime << endl;
#endif

  for (;!iter.done();iter++){
    IntVector c = *iter;
    Qavg[c] = ( Qvar[c] - Qvar_old[c] )/deltaTime;
  }
}


//______________________________________________________________________
//
template <class T>
void spatioTemporalAvg::query( const Patch         * patch,
                               constCCVariable< T >& Qvar,
                               CCVariable< T >     & Qavg,
                               CCVariable< T >     & Qvariance,
                               IntVector           & avgBoxCells,
                               CellIterator        & iter)
{
  IntVector lo = iter.begin();
  IntVector hi = iter.end();
  IntVector nBoxes(-9,-9,-9);

  for (int d=0; d<3; d++){
    nBoxes[d] = (int) std::ceil( (double) (hi[d] - lo[d])/(double)avgBoxCells[d] );
  }
  //__________________________________
  //  loop over boxes that this patch owns
  for ( int i=0; i<nBoxes.x(); i++ ){
    for ( int j=0; j<nBoxes.y(); j++ ){
      for ( int k=0; k<nBoxes.z(); k++ ){

        IntVector start = lo + IntVector(i,j,k) * avgBoxCells;
     //   cout << " Box: start: " << start << endl;

        // compute the average
        T Q_sum = T(0);
        double nCells=0;

        for(CellIterator aveCells( IntVector(0,0,0), avgBoxCells ); !aveCells.done(); aveCells++){
          IntVector c = start + *aveCells;
          if ( patch->containsIndex(lo, hi, c) ){
             Q_sum += Qvar[c];
             nCells++;
          }
        }

        T Q_avg  = Q_sum/T(nCells);
        T Q_avg2 = ( Q_sum * Q_sum )/T( nCells );
        T Q_var  = Q_avg2 - Q_avg * Q_avg;

        // Set the quantity
        for(CellIterator aveCells( IntVector(0,0,0), avgBoxCells ); !aveCells.done(); aveCells++){
          IntVector c = start + *aveCells;
          if ( patch->containsIndex(lo, hi, c) ){
            Qavg[c]      = Q_avg;
            Qvariance[c] = Q_var;
          }
        }
      }  // k box
    }  // j box
  }  // i box
}

//______________________________________________________________________
//  allocateAndZero  averaged
template <class T>
void spatioTemporalAvg::allocateAndZeroLabels( DataWarehouse * new_dw,
                                               const Patch   * patch,
                                               Qstats        & Q )
{
  int matl = Q.matl;
  allocateAndZero<T>( new_dw, Q.avgLabel,      matl, patch );
  allocateAndZero<T>( new_dw, Q.varianceLabel, matl, patch );
}


//______________________________________________________________________
//
int spatioTemporalAvg::getTimestep_OldDW()
{
  return d_baseTimestep;
}
