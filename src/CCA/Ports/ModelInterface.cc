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


#include <CCA/Ports/ModelInterface.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Output.h>

using namespace Uintah;

ModelInterface::ModelInterface(const ProcessorGroup* myworld,
			       const MaterialManagerP materialManager)
  : UintahParallelComponent(myworld), m_materialManager(materialManager)
{
}

ModelInterface::~ModelInterface()
{
}

void ModelInterface::setComponents( ApplicationInterface *comp )
{
  ApplicationInterface * parent = dynamic_cast<ApplicationInterface*>( comp );

  setAMR( parent->isAMR() );
  setDynamicRegridding( parent->isDynamicRegridding() );
  
  attachPort( "application", parent );
  attachPort( "scheduler",   parent->getScheduler() );
  attachPort( "regridder",   parent->getRegridder() );
  attachPort( "output",      parent->getOutput() );

  getComponents();
}

void ModelInterface::getComponents()
{
  m_application = dynamic_cast<ApplicationInterface*>( getPort("application") );

  if( !m_application ) {
    throw InternalError("dynamic_cast of 'm_application' failed!", __FILE__, __LINE__);
  }

  m_scheduler = dynamic_cast<Scheduler*>( getPort("scheduler") );

  if( !m_scheduler ) {
    throw InternalError("dynamic_cast of 'm_regridder' failed!", __FILE__, __LINE__);
  }

  m_regridder = dynamic_cast<Regridder*>( getPort("regridder") );

  if( isDynamicRegridding() && !m_regridder ) {
    throw InternalError("dynamic_cast of 'm_regridder' failed!", __FILE__, __LINE__);
  }

  m_output = dynamic_cast<Output*>( getPort("output") );

  if( !m_output ) {
    throw InternalError("dynamic_cast of 'm_output' failed!", __FILE__, __LINE__);
  }
}

void ModelInterface::releaseComponents()
{
  releasePort( "application" );
  releasePort( "scheduler" );
  releasePort( "regridder" );
  releasePort( "output" );

  m_application  = nullptr;
  m_scheduler    = nullptr;
  m_regridder    = nullptr;
  m_output       = nullptr;
}
