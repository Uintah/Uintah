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

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

using namespace Uintah;


int AnalysisModule::m_NUM_GRAPHS = 1;

//______________________________________________________________________
//
AnalysisModule::AnalysisModule( const ProcessorGroup* myworld,
                                const MaterialManagerP materialManager,
                                const ProblemSpecP& module_spec ) :
  UintahParallelComponent( myworld )
{
  m_materialManager = materialManager;
  m_module_spec = module_spec;

  // Time Step
  m_timeStepLabel = VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );
  
  // Simulation Time
  m_simulationTimeLabel = VarLabel::create(simTime_name, simTime_vartype::getTypeDescription() );

  // Delta t
  VarLabel* nonconstDelT =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
  nonconstDelT->allowMultipleComputes();
  m_delTLabel = nonconstDelT;
  
  
}
//______________________________________________________________________
//
AnalysisModule::~AnalysisModule()
{
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
  VarLabel::destroy(m_delTLabel);
}

//______________________________________________________________________
//    
void AnalysisModule::setComponents( ApplicationInterface *comp )
{
  ApplicationInterface * parent = dynamic_cast<ApplicationInterface*>( comp );

  attachPort( "application", parent );
  attachPort( "scheduler",   parent->getScheduler() );
  attachPort( "output",      parent->getOutput() );

  getComponents();
}
//______________________________________________________________________
//
void AnalysisModule::getComponents()
{
  m_application = dynamic_cast<ApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }

  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_scheduler' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }
}
//______________________________________________________________________
//
void AnalysisModule::releaseComponents()
{
  releasePort( "application" );
  releasePort( "scheduler" );
  releasePort( "output" );

  m_application  = nullptr;
  m_scheduler    = nullptr;
  m_output       = nullptr;
}

//______________________________________________________________________
/*
 Find the task graph index for each analysis module, which is needed if the module.
 is using temporal scheduling.

               This is tricky!!!  

 Modules may have unique sampling frequencies or may not.  If they're unique 
 then each module has it's own taskgraph compute index.
  
  Consider 4 modules with different analysis frequencies (f*)
  1) f1 = f2 = f3, f4    -> 2 Task graphs
  2) f1 = f4, f2 = f3    -> 2 Task graphs
  3) f1, f2, f3 = f4     -> 3 Task graphs
  3) f1, f2, f3, f4      -> 4 Task graphs
*/
//______________________________________________________________________

void
AnalysisModule::setTaskGraphIndex( std::vector<AnalysisModule*> &  modules)
{
  //Scheduler* sched = dynamic_cast<Scheduler*>( getPort("scheduler") );
  //int num_TG= sched->getNumTaskGraphs();
  
  // container filled with of the module analysis frequencies
  std::vector<double> freqs;
  
  for( size_t i=0; i!= modules.size(); i++){
    freqs.push_back( modules[i]->m_analysisFreq );
  }
  
  //__________________________________
  double ignoreMe = -9;
  int offset      = 1;           // this should be the number of TG
  int TG_index    = offset-1;
  
  //__________________________________
  //
  for( size_t i=0; i!= freqs.size(); i++){
    
    if (freqs[i] == ignoreMe){
      continue;
    }
    TG_index +=1;
    
    modules[i]->m_TG_computeIndex = TG_index;
    
    // if two module have identical analysis frequencies
    for( size_t j = i+1; j != freqs.size(); j++){

      if ( freqs[i] == freqs[j] ){
        modules[j]->m_TG_computeIndex = TG_index;
        freqs[j] = ignoreMe;
      }
    }
  }
  
  //__________________________________
  //  find number of unique task graphs
  int n_UniqueTG = 0;
  for( size_t i=0; i!= modules.size(); i++){
    int iTG = modules[i]->m_TG_computeIndex;
    std::cout << " OnTheFly AnalysisModule: " << i << " computeIndex: " << iTG  << std::endl;
    n_UniqueTG = std::max( n_UniqueTG, iTG );
    
  }
  m_NUM_GRAPHS = offset + n_UniqueTG;
  std::cout << " OnTheFly AnalysisModule: num TaskGraphs: " << m_NUM_GRAPHS << std::endl;
  
}

//______________________________________________________________________
//
void AnalysisModule::sched_TimeVars( Task* t,
                                     const LevelP   & level,
                                     const VarLabel * prev_AnlysTimeLabel,
                                     const bool addComputes )
{
  t->requires( Task::OldDW, m_simulationTimeLabel );
  t->requires( Task::OldDW, prev_AnlysTimeLabel );
  t->requires( Task::OldDW, m_delTLabel, level.get_rep() );
  
  if( addComputes ){
    t->computes( prev_AnlysTimeLabel );
  }
}

//______________________________________________________________________
//
bool AnalysisModule::getTimeVars( DataWarehouse  * old_dw,
                               const VarLabel * prev_AnlysTimeLabel,
                               timeVars       & tv)
{
  max_vartype     prevTime;
  simTime_vartype simTime;
  delt_vartype    delT;

  old_dw->get( prevTime, prev_AnlysTimeLabel );
  old_dw->get( simTime,  m_simulationTimeLabel );
  old_dw->get( delT,     m_delTLabel );

  tv.prevAnlysTime = prevTime;
  tv.nextAnlysTime = prevTime + 1.0/m_analysisFreq;
  tv.now           = simTime + delT;

  if( tv.now < d_startTime || tv.now > d_stopTime || tv.now < tv.nextAnlysTime ){
    tv.isItTime = false;
    return false;
  } else {
    tv.isItTime = true;
  }

  return tv.isItTime;  
}
//______________________________________________________________________
//
void AnalysisModule::putTimeVars( DataWarehouse  * new_dw,
                                  const VarLabel * prev_AnlysTimeLabel,
                                  timeVars tv)
{
  if( tv.isItTime ){
    new_dw->put(max_vartype( tv.nextAnlysTime ), prev_AnlysTimeLabel);
  } 
  else {
    new_dw->put(max_vartype( tv.prevAnlysTime ), prev_AnlysTimeLabel);
  }
}

//______________________________________________________________________
//
bool AnalysisModule::isItTime( DataWarehouse * old_dw,
                              const VarLabel * prev_AnlysTimeLabel)
{
  timeVars tv;
  return getTimeVars( old_dw, prev_AnlysTimeLabel, tv);
}
