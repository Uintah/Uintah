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
 *  PlumeFrameworkImpl.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 */

#ifndef SCIRun_Framework_PlumeFrameworkImpl_h
#define SCIRun_Framework_PlumeFrameworkImpl_h


#include <SCIRun/Distributed/DistributedFrameworkInternal.h>
#include <SCIRun/Plume/CCAComponentModel.h>
#include <SCIRun/Distributed/DistributedFramework.h>

namespace SCIRun {

  namespace Plume = sci::cca::plume;
  /**
   * \class PlumeFrameworkImpl
   * 
   * \brief An implementation of a PlumeFrameworkImpl 
   */
  
  template<class Base>
  class PlumeFrameworkImpl : public DistributedFrameworkInternal<Base>
  {
  public:
    typedef Plume::PlumeFramework::pointer pointer;
    
    PlumeFrameworkImpl( const DistributedFramework::internalPointer &parent = 
			DistributedFramework::internalPointer(0));
    virtual ~PlumeFrameworkImpl();
    
    /*
     * Two pure virtual methods to create and destroy a component.
     */
    virtual Distributed::ComponentInfo::pointer
    createComponent( const std::string &, const std::string &, const sci::cca::TypeMap::pointer &);

    virtual void destroyComponent( const Distributed::ComponentInfo::pointer &info);

    /*
     * from AbstractFramework
     */

    /** */
    virtual sci::cca::TypeMap::pointer createTypeMap();

    /** */
    virtual sci::cca::Services::pointer getServices( const std::string &, 
						     const std::string &, 
						     const sci::cca::TypeMap::pointer &);
    
    /** */
    virtual void releaseServices( const sci::cca::Services::pointer &);

    /** */
    virtual void shutdownFramework();

    /** */
    virtual sci::cca::AbstractFramework::pointer createEmptyFramework();

  protected:
    CCAComponentModel cca;
    
};
  
} // end namespace SCIRun

//#include <SCIRun/Plume/PlumeFrameworkImpl.code>

#endif
