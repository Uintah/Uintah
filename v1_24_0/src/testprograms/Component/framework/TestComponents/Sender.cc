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


#include <testprograms/Component/framework/TestComponents/Sender.h>
#include <testprograms/Component/framework/PortInfoImpl.h>

#include <sstream>
#include <iostream>
using namespace std;

namespace sci_cca {

using std::cerr;

Sender::Sender()
{
}

Sender::~Sender()
{
  cout << "~Sender()\n";
}

void 
Sender::setServices( const Services::pointer &svc )
{
  ComponentImpl::setServices( svc );

  if( !svc.isNull() )
    {
      PortInfo::pointer info (new PortInfoImpl("Uses", "", 0));
      svc->registerUsesPort( info );
    }
}

void
Sender::go()
{
  TestPort::pointer port (pidl_cast<TestPort::pointer>(services_->getPort("Uses")));
  if ( port.isNull() ) {
    cerr << "go: could not find test port\n";
    return;
  }

  port->test();
}

} // namespace sci_cca

