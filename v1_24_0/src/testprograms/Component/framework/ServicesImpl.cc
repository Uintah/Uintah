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

#include <testprograms/Component/framework/ServicesImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <iostream>

using std::cerr;

namespace sci_cca {

ServicesImpl::ServicesImpl() 
{
}


ServicesImpl::~ServicesImpl()
{
  cerr << "Services " << id_ << " exit\n";
}

Port
ServicesImpl::getPort( const string & name )
{
  return 0;
}

PortInfo
ServicesImpl::createPortInfo( const string & name,
			      const string & type,
			      const array1<string> & properties )
{
  return new PortInfoImpl( name, type, properties );
}

void
ServicesImpl::registerUsesPort( const PortInfo & nameAndType )
{
}

void
ServicesImpl::unregisterUsesPort( const string & name )
{
}

void   
ServicesImpl::addProvidesPort( const Port & inPort,
		 	       const PortInfo & name )
{
  cerr << "ServicesImpl: addProvidesPort  not implemented yet\n";
//   throw InternalError( (string("addProvidesPort for: ") 
// 		       + " is not implemented yet.").c_str());

}

void
ServicesImpl::removeProvidesPort( const string & name )
{
  cerr << "removeProvidesPort not implemented yet\n";
  throw InternalError((string("removeProvidesPort for: ") + name
		       + " is not implemented yet.").c_str());
}

void
ServicesImpl::releasePort( const string & name )
{
  cerr << "releasePort not implemented yet\n";
  throw InternalError( (string("releasePort for: ") + name
			+ " is not implemented yet.").c_str() );
}
 
ComponentID
ServicesImpl::getComponentID()
{
  return id_;
}


} // namespace sci_cca
