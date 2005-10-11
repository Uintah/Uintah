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
 *  FrameworkPropertiesService.h: Implementation of the CCA FrameworkPropertiesService interface for SCIRun
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_FrameworkPropertiesService_h
#define SCIRun_FrameworkPropertiesService_h

#include <SCIRun/Distributed/FrameworkPropertiesServiceImpl.h>
#include <vector>

namespace SCIRun {

  class DistributedFramework;
  class FrameworkProperties;
  
  /**
   * \class FrameworkPropertiesService
   *
   */
  class FrameworkPropertiesService : 
    public FrameworkPropertiesServiceImpl<Distributed::ports::FrameworkPropertiesService>
  {
  public:
    typedef Distributed::ports::FrameworkPropertiesService Interface;
    typedef Distributed::ports::FrameworkPropertiesService::pointer pointer;
    
    FrameworkPropertiesService(const DistributedFramework::internalPointer &framework)
      : FrameworkPropertiesServiceImpl<Interface>(framework)
    {}
    
    virtual ~FrameworkPropertiesService();
    
    /** Factory method for allocating new FrameworkPropertiesService objects.  Returns
	a smart pointer to the newly allocated object registered in the framework
	\em fwk with the instance name \em name. */
    
    static pointer create(const DistributedFramework::internalPointer &framework);
    
  private:
    DistributedFramework *framework;
    
  };


} // end namespace SCIRun

#endif
