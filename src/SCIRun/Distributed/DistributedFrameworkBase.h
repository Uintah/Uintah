/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2005 Scientific Computing and Imaging Institute,
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
 *  DistributedFrameworkBase.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_Distributed_DistributedFrameworkBase_h
#define SCIRun_Distributed_DistributedFrameworkBase_h

#include <list>
#include <Core/Thread/Mutex.h>
#include <SCIRun/Core/CoreFrameworkBase.h>

namespace SCIRun {

  using namespace sci::cca;
  using namespace sci::cca::core;
  using namespace sci::cca::distributed;

  /**
   * \class DistributedFramework
   * 
   * \brief An implementation of a DistributedFramework 
   */


  template<class Interface>
  class DistributedFrameworkBase : public CoreFrameworkBase<Interface>
  {
  public:
    typedef DistributedFramework::pointer pointer;

    DistributedFrameworkBase( const DistributedFramework::pointer &parent = 0);
    virtual ~DistributedFrameworkBase();

    /*
     * methods that implement the DistributedFramework 
     */

    virtual bool isRoot() { return parent.isNull(); }
    virtual DistributedFramework::pointer getParent() { return parent; }

    virtual FrameworkID::pointer getFrameworkID();
    virtual SSIDL::array1<DistributedFramework::pointer> getChildren();

  private:
    typedef std::list<DistributedFramework::pointer> ChildrenList;

    DistributedFramework::pointer parent;
    ChildrenList children;

    Mutex children_lock;
  };
  
} // end namespace SCIRun


#endif
