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
 *  InternalComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/Internal/InternalComponentInstance.h>
#include <SCIRun/PortInstance.h>
#include <iostream>

namespace SCIRun {

InternalComponentInstance::InternalComponentInstance(
    SCIRunFramework* framework,
    const std::string& instanceName,
    const std::string& className)
  : ComponentInstance(framework, instanceName, className,
                      sci::cca::TypeMap::pointer(0)), useCount(0)
{
}

InternalComponentInstance::~InternalComponentInstance()
{
}

PortInstance::pointer
InternalComponentInstance::getPortInstance(const std::string& /*name*/)
{
  std::cerr << "InternalComponentInstance::getPortInstance not finished"
            << std::endl;
  return PortInstance::pointer(0);
}

PortInstanceIterator* InternalComponentInstance::getPorts()
{
  std::cerr << "SCIRunComponentInstance::getPorts not finished!"
            << std::endl;
  return 0;
}

void InternalComponentInstance::incrementUseCount()
{
  useCount++;
}

bool InternalComponentInstance::decrementUseCount()
{
  if(useCount<=0)
    return false;
  
  useCount--;
  return true;
}

} // end namespace SCIRun
