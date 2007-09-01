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
 *  ComponentInstance.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Framework/Core/ComponentInstance.h>
#include <iostream>

using namespace scijump;

namespace SCIRun {

ComponentInstance::ComponentInstance(SCIJumpFramework* framework,
                                     const std::string &instanceName,
                                     const std::string &className,
                                     const gov::cca::TypeMap &tm)
    : framework(framework), instanceName(instanceName),
      className(className), properties(tm), releaseCallback(0)
{
  if (properties._is_nil()) {
    properties = framework->createTypeMap();
  }
  // cca.className is a CCA standardized key (see BuilderService documentation):
  properties.putString("cca.className", className);
  properties.putString("cca.instanceName", instanceName);
}

ComponentInstance::~ComponentInstance()
{
}

void
ComponentInstance::setComponentProperties(const gov::cca::TypeMap &tm)
{
  // TODO: check properties - do not allow cca.className to be changed
  properties = tm;
}

bool
ComponentInstance::releaseComponentCallback(const gov::cca::Services &svc)
{
  if (releaseCallback._is_nil()) return false;

  releaseCallback.releaseServices(svc);
  return true;
}


} // end namespace SCIRun
