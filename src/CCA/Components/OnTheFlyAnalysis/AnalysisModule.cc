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

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

// NOTE: UintahParallelComponent is noramlly called with the ProcessorGroup

AnalysisModule::AnalysisModule( const ProcessorGroup* myworld,
                                const MaterialManagerP materialManager,
                                const ProblemSpecP& module_spec ) :
  UintahParallelComponent( myworld )
{
  m_materialManager = materialManager;
  m_module_spec = module_spec;

  // Time Step
  m_timeStepLabel =
    VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );
  
  // Simulation Time
  m_simulationTimeLabel =
    VarLabel::create(simTime_name, simTime_vartype::getTypeDescription() );

  // Delta t
  VarLabel* nonconstDelT =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
  nonconstDelT->allowMultipleComputes();
  m_delTLabel = nonconstDelT;
}

AnalysisModule::~AnalysisModule()
{
  VarLabel::destroy(m_timeStepLabel);
  VarLabel::destroy(m_simulationTimeLabel);
  VarLabel::destroy(m_delTLabel);
}
    
void AnalysisModule::setComponents( ApplicationInterface *comp )
{
  ApplicationInterface * parent = dynamic_cast<ApplicationInterface*>( comp );

  attachPort( "application", parent );
  attachPort( "scheduler",   parent->getScheduler() );
  attachPort( "output",      parent->getOutput() );

  getComponents();
}

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

void AnalysisModule::releaseComponents()
{
  releasePort( "application" );
  releasePort( "scheduler" );
  releasePort( "output" );

  m_application  = nullptr;
  m_scheduler    = nullptr;
  m_output       = nullptr;
}

