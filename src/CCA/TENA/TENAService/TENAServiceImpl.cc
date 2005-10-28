/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  TENAServiceImpl.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institue
 *   University of Utah
 *   October 2005
 *
 */

#include <Core/CCA/spec/sci_sidl.h>
#include <Core/Thread/Guard.h>

#include <TENA/Middleware/config.h>
#include <CCA/TENA/TENAService/TENAServiceImpl.h>
#include <CCA/TENA/TENAService/ExecutionImpl.h>

namespace SCIRun {
  
  using namespace sci::cca;
  using namespace sci::cca::core;
  using namespace sci::cca::tena;

  struct TENAServiceInfo {
    TENAServiceInfo() {}

    TENA::Middleware::RuntimePtr runtime;
    TENA::Middleware::Configuration configuration;
  };

  TENAServiceImpl::TENAServiceImpl()
    : initialized(false), lock("TENAServiceImpl::lock")
  {
    info = new TENAServiceInfo;
  }

  TENAServiceImpl::~TENAServiceImpl()
  {
    // FIXME [yarden]: do we need to inform the TENA middleware ?
    delete info;
  }
  
  bool TENAServiceImpl::setConfiguration()
  {
    // FIXME [yarden]: not implemented yet.
    return false;
  }

  Execution::pointer TENAServiceImpl::joinExecution(const std::string &name)
  {
    Guard guard(&lock);

    if (!initialized) {
      info->runtime = TENA::Middleware::init(info->configuration);
      initialized = true;
    }
    
    ExecutionsMap::iterator iter = executions.find(name);
    if ( iter != executions.end() )
      return iter->second;

    Execution::pointer exec = new ExecutionImpl( info->runtime->joinExecution(name) );
    executions[name] = exec;

    return exec;
  }
  
} // end namespace SCIRun

