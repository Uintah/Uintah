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
 *  PortInfo.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   Sept 2005
 *
 */
#include <SCIRun/Distributed/PortInfo.h>
#include <SCIRun/Distributed/PortInfoImpl.code>

namespace SCIRun {
  
  PortInfo::PortInfo(const std::string& portname, 
		     const std::string& classname,
		     const sci::cca::TypeMap::pointer& properties,
		     Distributed::PortType porttype)
    : PortInfoImpl<Distributed::PortInfo>( portname, classname, properties, porttype )
  {
  }
  
  PortInfo::PortInfo(const std::string& portname, 
		     const std::string& classname,
		     const sci::cca::TypeMap::pointer& properties,
		     const sci::cca::Port::pointer& port,
		     Distributed::PortType porttype)
    : PortInfoImpl<Distributed::PortInfo>( portname, classname, properties, port, porttype)
  {
  }

} // end namespace SCIRun

