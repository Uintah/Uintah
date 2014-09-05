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


#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/cca_sidl.h>

#include <Core/Exceptions/InternalError.h> 
#include <Core/Thread/CrowdMonitor.h>

#include <iostream>


namespace sci_cca {

using namespace SCIRun;
using namespace std;

ComponentRecord::ComponentRecord( const ComponentID::pointer &id )
{
  id_ = id;
}

ComponentRecord::~ComponentRecord()
{
}

Port::pointer
ComponentRecord::getPort( const string &name )
{
  use_iterator ui = uses_.find(name);
  if ( ui == uses_.end() ) {
    // throw an error ? the port was not registered as a use port.
    return Port::pointer(0);
  }

  UsePortRecord *use = ui->second;
  
  if ( !use->connection_ ) {
    // a non blocking implementation
    return Port::pointer(0);
  }
  
  ProvidePortRecord *provide = use->connection_->provide_;
  if ( !provide ) 
    throw InternalError( "Provide port is missing...", __FILE__, __LINE__ );

  return provide->port_;
}
    
void 
ComponentRecord::registerUsesPort( const PortInfo::pointer &info) 
{
  UsePortRecord *record = new UsePortRecord;
  record->info_ = info;
  record->connection_ = 0;

  uses_[info->getName()] = record;
}


void 
ComponentRecord::unregisterUsesPort( const string &name )
{
  use_iterator ui = uses_.find( name );
  if ( ui != uses_.end() ) {
    // check if the use port is connected. 
    if ( ui->second->connection_ ) {
      // if so than disconnect it
      ui->second->connection_->disconnect();
    }
    delete ui->second;
    uses_.erase(ui);
  }
  else {
    // should it report an error if port was not found ?
  }
}

void 
ComponentRecord::addProvidesPort( const Port::pointer &port, const PortInfo::pointer& info) 
{
  ProvidePortRecord *record = new ProvidePortRecord;
  record->info_ = info;
  record->connection_ = 0;
  record->port_ = port;
  record->in_use_ = false;

  provides_[info->getName()] = record;
}

void 
ComponentRecord::removeProvidesPort( const string &name)
{
  provide_iterator pi = provides_.find(name);
  if ( pi != provides_.end() ) {
    if ( pi->second->connection_ ) {
      pi->second->connection_->disconnect();
    }
    delete pi->second;
    provides_.erase(pi);
  }
  else {
    // should it report an error if port was not found ?
  }
}

void 
ComponentRecord::releasePort( const string &name)
{
  use_iterator ui = uses_.find(name);
  if ( ui == uses_.end() ) {
    throw InternalError( "Release port does not exist", __FILE__, __LINE__ );
  }

  UsePortRecord *record = ui->second;

  if ( record->connection_ ) {
    record->connection_->disconnect();
    record->connection_ = 0;
  } else {
    // is this an error worthy of an exception ?
  }
}



ProvidePortRecord *
ComponentRecord::getProvideRecord( const string &name )
{
  provide_iterator pi = provides_.find(name);
  if ( pi != provides_.end() ) {
    return pi->second;
  }
  
  cerr << "CR:: provide port not found\n";
  return 0;
  
}
UsePortRecord *
ComponentRecord::getUseRecord( const string &name )
{
  use_iterator ui = uses_.find(name);
  if ( ui != uses_.end() ) { 
    return ui->second;
  }
  
  cerr << "CR::" << id_->toString() << " use port ("<<name<<") not found\n";
  return 0;
  
}


void
ConnectionRecord::disconnect()
{
  // disconnect connection
  
  // clean up
  use_->connection_ = 0;
  provide_->connection_ = 0;
  delete this;
}


Registry::Registry() : connections_("Registry Connection Lock") 
{
}

ProvidePortRecord *
Registry::getProvideRecord( const ComponentID::pointer &id, const string &name )
{
  component_iterator from = components_.find(id->toString());
  if ( from == components_.end() ) {
    cerr << "Registry::component not found\n";
    return 0; 
  }
  else {
    return from->second->getProvideRecord(name);
  }
}

UsePortRecord *
Registry::getUseRecord( const ComponentID::pointer &id, const string &name )
{
  component_iterator from = components_.find(id->toString());
  if ( from == components_.end() ) {
    cerr << "Reg: component not found\n";
    return 0; 
  }
  else {
    return from->second->getUseRecord(name);
  }
}

} // namespace sci_cca

