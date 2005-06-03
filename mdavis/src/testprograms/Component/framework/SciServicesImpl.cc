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

#include <testprograms/Component/framework/SciServicesImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

SciServicesImpl::SciServicesImpl() 
{
}

SciServicesImpl::~SciServicesImpl()
{
  cerr << "SciServices " << id_.getPointer() << " exit\n";
}

Port::pointer
SciServicesImpl::getPort( const string &name )
{
  return framework_->getPort( id_, name );
}

PortInfo::pointer
SciServicesImpl::createPortInfo( const string &name,
			      const string &type,
			      const array1<string> &properties )
{
  return PortInfo::pointer(new PortInfoImpl( name, type, properties ));
}

void
SciServicesImpl::registerUsesPort( const PortInfo::pointer &port_info )
{
  framework_->registerUsesPort( id_, port_info );
}

void
SciServicesImpl::unregisterUsesPort( const string &name )
{
  framework_->unregisterUsesPort( id_, name );
}

void   
SciServicesImpl::addProvidesPort( const Port::pointer &port,
				  const PortInfo::pointer &info )
{
  framework_->addProvidesPort( id_, port, info );
}

void
SciServicesImpl::removeProvidesPort( const string & name )
{
  framework_->removeProvidesPort( id_, name );
}

void
SciServicesImpl::releasePort( const string & name )
{
  framework_->releasePort( id_, name );
}
 
ComponentID::pointer
SciServicesImpl::getComponentID()
{
  return id_;
}

void 
SciServicesImpl::init( const Framework::pointer &f,
		       const ComponentID::pointer & id) 
{ 
  framework_ = f; 
  id_ = id; 
}

void
SciServicesImpl::done()
{
  framework_ = 0;
  id_ = 0;
}

void 
SciServicesImpl::shutdown() 
{ 
  framework_->unregisterComponent( id_); 
}

} // namespace sci_cca
