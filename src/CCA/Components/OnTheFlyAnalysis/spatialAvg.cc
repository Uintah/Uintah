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

#include <CCA/Components/OnTheFlyAnalysis/spatialAvg.h>
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
    <Module type = "spatialAvg">
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
using namespace std;

Dout dout_OTF_spatialAvg("spatialAvg",     "OnTheFlyAnalysis", "Task scheduling and execution.", false);

//______________________________________________________________________
spatialAvg::spatialAvg( const ProcessorGroup*  myworld,
                        const MaterialManagerP materialManager,
                        const ProblemSpecP&    module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{}

//______________________________________________________________________
//
spatialAvg::~spatialAvg()
{

  DOUTR(dout_OTF_spatialAvg, "Doing destructor statistics");

  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }

  // delete each QavgVar label
  for (unsigned int i =0 ; i < d_QavgVars.size(); i++) {
    QavgVar& Q = d_QavgVars[i];
    VarLabel::destroy( Q.avgLabel );
    VarLabel::destroy( Q.varianceLabel );
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void spatialAvg::problemSetup(const ProblemSpecP &,
                              const ProblemSpecP & restart_prob_spec,
                              GridP & grid,
                              std::vector<std::vector<const VarLabel* > > &PState,
                              std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  DOUTR(dout_OTF_spatialAvg, "Doing spatialAvg::problemSetup");


  int numMatls  = m_materialManager->getNumMatls();

  proc0cout << "__________________________________Data Analysis module: spatialAvg" << endl;
  proc0cout << "         Computing spatial average for all of the variables listed"<< endl;

  //__________________________________
  //  region to average over
  m_module_spec->require("avgBoxCells", d_avgBoxCells );

  //__________________________________
  //  domain
  string ans = "null";
  m_module_spec->get( "domain", ans );

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
  m_module_spec->require("timeStart",  d_startTime);
  m_module_spec->require("timeStop",   d_stopTime);

  // Start time < stop time
  if(d_startTime > d_stopTime ){
    throw ProblemSetupException("\n ERROR:spatialAvg: startTime > stopTime. \n", __FILE__, __LINE__);
  }
  
  proc0cout << "  StartTime: " << d_startTime << " stopTime: "<< d_stopTime << endl;

  // debugging
  m_module_spec->get("monitorCell", d_monitorCell);

  //__________________________________
  // find the material to extract data from.
  Material* matl = nullptr;

  if(m_module_spec->findBlock("material") ){
    matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }
  else {
    throw ProblemSetupException("ERROR:AnalysisModule:spatialAvg: Missing <material> tag. \n", __FILE__, __LINE__);
  }

  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  //__________________________________
  //  Read in variables label names
   ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("spatialAvg: Couldn't find <Variables> tag", __FILE__, __LINE__);
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
      throw ProblemSetupException("spatialAvg: problemSetup: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);
    
    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find( name, "ERROR spatialAvg::problemSetup <analyze>");

    //__________________________________
    // Only CCVariable doubles, floats and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if( td->getType() != TypeDescription::CCVariable  ||
        ( subtype->getType() != TypeDescription::double_type &&
          subtype->getType() != TypeDescription::float_type &&
          subtype->getType() != TypeDescription::Vector ) ) {
      ostringstream warn;
      warn << "ERROR:Module:spatialAvg: ("<<label->getName() << " " << td->getName() << " ) has not been implemented\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    QavgVar Q;
    Q.matl    = matl;
    Q.Q_Label = label;
    Q.subtype = subtype;

    Q.avgLabel      = VarLabel::create( "spatialAvg_" + name,     td);
    Q.varianceLabel = VarLabel::create( "spatialAvg_variance_" + name, td);
    d_QavgVars.push_back( Q );

    //__________________________________
    //  bulletproofing
    std::string variance = "spatialAvg_variance_"+ name;
    ostringstream mesg;
    mesg << "";
    if( !m_output->isLabelSaved( variance ) ){
      mesg << variance;
    }

    if( mesg.str() != "" ){
      ostringstream warn;
      warn << "WARNING:  You've activated the  AnalysisModule:spatialAvg module but your not saving the variable(s) ("
           << mesg.str() << ")";
      proc0cout << warn.str() << endl;
    }
  }
  
  //__________________________________
  //  create the matl set
  d_matlSet = scinew MaterialSet();
  d_matlSet->addAll_unique(m);
  d_matlSet->addReference();
  d_matSubSet = d_matlSet->getUnion();
  proc0cout << "__________________________________ Data Analysis module: spatialAvg" << endl;

}

//______________________________________________________________________
void spatialAvg::scheduleInitialize(SchedulerP   & sched,
                                    const LevelP & level)
{
  printSchedule( level, dout_OTF_spatialAvg, "spatialAvg::scheduleInitialize" );

  Task* t = scinew Task("spatialAvg::initialize",
                   this,&spatialAvg::initialize);

  for ( unsigned int i =0 ; i < d_QavgVars.size(); i++ ) {
    const QavgVar Q = d_QavgVars[i];
    t->computes ( Q.avgLabel );
    t->computes ( Q.varianceLabel );
  }
  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
//
void spatialAvg::initialize(const ProcessorGroup  *,
                            const PatchSubset     * patches,
                            const MaterialSubset  *,
                            DataWarehouse         *,
                            DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,dout_OTF_spatialAvg,"Doing spatialAvg::initialize");

    for ( unsigned int i =0 ; i < d_QavgVars.size(); i++ ) {
      QavgVar& Q = d_QavgVars[i];

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
          throw InternalError("spatialAvg: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstat
  }  // pathes
}

//______________________________________________________________________
//
void spatialAvg::scheduleDoAnalysis(SchedulerP   & sched,
                                    const LevelP & level)
{
  printSchedule( level,dout_OTF_spatialAvg,"spatialAvg::scheduleDoAnalysis" );

  Task* t = scinew Task("spatialAvg::doAnalysis",
                   this,&spatialAvg::doAnalysis);

  t->requires( Task::OldDW, m_timeStepLabel);
  t->requires( Task::OldDW, m_simulationTimeLabel);

  Ghost::GhostType  gn  = Ghost::None;

  for ( unsigned int i =0 ; i < d_QavgVars.size(); i++ ) {
    QavgVar Q = d_QavgVars[i];

    // define the matl subset for this variable
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( Q.matl );
    matSubSet->addReference();

    t->requires( Task::NewDW, Q.Q_Label, matSubSet, gn, 0 );
    t->computes ( Q.avgLabel,       matSubSet );
    t->computes ( Q.varianceLabel,  matSubSet );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
//    Q.print();
  }
  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
// Compute the spatialAvg for each variable the user requested
void spatialAvg::doAnalysis(const ProcessorGroup * pg,
                            const PatchSubset    * patches,
                            const MaterialSubset * ,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dout_OTF_spatialAvg, "Doing spatialAvg::doAnalysis");

    for ( unsigned int i =0 ; i < d_QavgVars.size(); i++ ) {
      QavgVar& Q = d_QavgVars[i];

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
          throw InternalError("spatialAvg: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // QavgVar loop
  }  // patches
}

//______________________________________________________________________
//  computeAvgWrapper:
template <class T>
void spatialAvg::computeAvgWrapper( DataWarehouse     * old_dw,
                                    DataWarehouse     * new_dw,         
                                    const PatchSubset * patches,        
                                    const Patch       * patch,          
                                    QavgVar& Q)                          
{
  simTime_vartype simTime;
  old_dw->get( simTime, m_simulationTimeLabel );
  
  if(simTime < d_startTime || simTime > d_stopTime){
    //proc0cout << " IGNORING------------DataAnalysis: spatialAvg" << endl;
    allocateAndZeroLabels< T >( new_dw, patch, Q );
  }
  else {
    //proc0cout << " Computing------------DataAnalysis: spatialAvg" << endl;

    computeAvg< T >(old_dw, new_dw, patch, Q);
  }
}



//______________________________________________________________________
//
template <class T>
void spatialAvg::computeAvg( DataWarehouse  * old_dw,
                             DataWarehouse  * new_dw,
                             const Patch    * patch,
                             QavgVar& Q)
{
  timeStep_vartype timeStep;
  old_dw->get( timeStep, m_timeStepLabel );
  timeStep = timeStep -1;  
  
  static proc0patch0cout mesg( d_QavgVars.size() );
  ostringstream msg;
  msg <<"    spatialAvg::computeAvg( "<< Q.Q_Label->getName() << " )\n";
  mesg.print(patch, msg );
  
  //__________________________________
  
  const int matl = Q.matl;
  
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
    spatialAvg::query( patch, Qvar, Qavg, Qvariance, d_avgBoxCells, iter);
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

      spatialAvg::query( patch, Qvar, Qavg, Qvariance, avgPanelCells, iter);

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
void spatialAvg::query( const Patch         * patch,
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
void spatialAvg::allocateAndZeroLabels( DataWarehouse * new_dw,
                                               const Patch   * patch,
                                               QavgVar        & Q )
{
  int matl = Q.matl;
  allocateAndZero<T>( new_dw, Q.avgLabel,      matl, patch );
  allocateAndZero<T>( new_dw, Q.varianceLabel, matl, patch );
}

//______________________________________________________________________
//  allocateAndZero
template <class T>
void spatialAvg::allocateAndZero( DataWarehouse * new_dw,
                              const VarLabel * label,
                              const int matl,
                              const Patch * patch )
{
  CCVariable<T> Q;
  new_dw->allocateAndPut( Q, label, matl, patch );
  T zero(0.0);
  Q.initialize( zero );
}

//______________________________________________________________________
// Instantiate the explicit template instantiations.
//
template void spatialAvg::allocateAndZero<float> ( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );
template void spatialAvg::allocateAndZero<double>( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );
template void spatialAvg::allocateAndZero<Vector>( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );

//______________________________________________________________________
//
spatialAvg::proc0patch0cout::proc0patch0cout( const int nPerTimestep)
{
  d_nTimesPerTimestep = nPerTimestep;
}

//______________________________________________________________________
//
void spatialAvg::proc0patch0cout::print(const Patch * patch,
                                   std::ostringstream& msg)
{
  if( d_count <= d_nTimesPerTimestep ){
    if( patch->getID() == 0 && Uintah::Parallel::getMPIRank() == 0 && Uintah::Parallel::getMainThreadID() == std::this_thread::get_id() ){
      std::cout << msg.str();
      d_count += 1;  
    }
  } 
  else {
    d_count = 0;
  }
}
