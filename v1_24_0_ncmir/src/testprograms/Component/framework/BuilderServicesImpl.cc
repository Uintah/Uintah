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

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/BuilderServicesImpl.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

#include <iostream>

namespace sci_cca {

using namespace std;

typedef Registry::component_iterator component_iterator;

BuilderServicesImpl::BuilderServicesImpl() 
{
}


BuilderServicesImpl::~BuilderServicesImpl()
{
}


void 
BuilderServicesImpl::init( const Framework::pointer &f ) 
{ 
  framework_ = f; 
  
  registry_ = dynamic_cast<FrameworkImpl *>(f.getPointer())->registry_;
}

bool
BuilderServicesImpl::connect( const ComponentID::pointer &uses, 
				 const string &use_port, 
				 const ComponentID::pointer &provider, 
				 const string &provide_port)
{
  // lock registry
  registry_->connections_.writeLock();

  // get provide port record
  ProvidePortRecord *provide = registry_->getProvideRecord( provider, 
							    provide_port );
  if ( !provide ) {
    // error: could not find provider's port
    cerr <<"provide not found\n";
    return false;
  }

  if ( provide->connection_ ) {
    // error: provide port in use
    cerr << "provide port in use\n";
    return false;
  }

  // get use port record
  cerr << "connections: uses id is " << uses->toString() << endl;
  UsePortRecord *use = registry_->getUseRecord( uses, use_port );

  if ( !use ) {
    // error: could not find use's port
    cerr << "uses not found\n";
    return false;
  }

  if ( use->connection_ ) {
    // error: uses port in use
    cerr << "use connection in use\n";
    return false;
  }

  // connect
  ConnectionRecord *record = new ConnectionRecord;
  record->use_ = use;
  record->provide_ = provide;

  provide->connection_ = record;
  use->connection_ = record;

  // unlock registry
  registry_->connections_.writeUnlock();

  // notify who ever wanted to 

  // done
  return true;
}
  

bool 
BuilderServicesImpl::disconnect( const ComponentID::pointer &, const string &, 
				    const ComponentID::pointer &, const string &)
{
  return false;
}

bool 
BuilderServicesImpl::exportAs( const ComponentID::pointer &, const string &, 
				  const string &)
{
  return false;
}

bool
BuilderServicesImpl::provideTo( const ComponentID::pointer &, const string&, 
				   const string &)
{
  return false;
}

} // namespace sci_cca
