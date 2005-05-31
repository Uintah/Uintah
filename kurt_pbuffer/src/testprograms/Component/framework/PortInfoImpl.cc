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

#include <testprograms/Component/framework/PortInfoImpl.h>

namespace sci_cca {

PortInfoImpl::PortInfoImpl( const string & name, 
			    const string & type, 
			    const array1<string> & properties )
  : name_(name), type_(type), properties_(properties)
{
}

PortInfoImpl::~PortInfoImpl()
{
}

string
PortInfoImpl::getType()
{
  return type_;
}

string
PortInfoImpl::getName()
{
  return name_;
}

string
PortInfoImpl::getProperty( const string & name )
{
  // Dd: Why is this +2?
  for( unsigned int i=0; i < properties_.size(); i += 2 ) {
    if( properties_[i] == name ) 
      return properties_[i+1];
  }

  return "";
}

} // namespace sci_cca
