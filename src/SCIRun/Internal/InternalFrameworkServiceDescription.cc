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
 *  InternalFrameworkServiceDescription.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#include <SCIRun/SCIRunFramework.h>
#include <SCIRun/Internal/InternalFrameworkServiceDescription.h>
#include <SCIRun/Internal/InternalFrameworkServiceInstance.h>
#include <SCIRun/Internal/InternalComponentModel.h>
#include <SCIRun/Internal/FrameworkInternalException.h>

#include <iostream>

namespace SCIRun {

  InternalFrameworkServiceDescription::InternalFrameworkServiceDescription(
        InternalComponentModel* model,
	const std::string& serviceType,
	InternalFrameworkServiceInstance* (*create)(SCIRunFramework*))
    : InternalServiceDescription(model, serviceType), create(create), singleton_instance(0)
{
}
  
  InternalFrameworkServiceDescription::~InternalFrameworkServiceDescription()
  {
    std::cerr << "What if singleton_instance is refcounted?" << std::endl;
    if(singleton_instance) {
      delete singleton_instance;
    }
  }

  InternalFrameworkServiceInstance *InternalFrameworkServiceDescription::get(SCIRunFramework *fwk) 
  {
    if ( singleton_instance == 0 ) {
      singleton_instance = create(fwk);
      fwk->registerComponent(singleton_instance, singleton_instance->getInstanceName());
    }
    return singleton_instance;
  }

  void InternalFrameworkServiceDescription::release(SCIRunFramework *fwk)
  {
    if ( singleton_instance == 0 ) return;

    singleton_instance->decrementUseCount();
//     if ( singleton_instance->decrementUseCount() ) 
//       throw FrameworkInternalException("service ["+singleton_instance->getInstanceName()+"] released too many times");
  }

} // end namespace SCIRun
