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
 *  CCAComponentClassDescriptionImpl.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_CCAComponentClassDescriptionImpl_h
#define SCIRun_CCAComponentClassDescriptionImpl_h

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Distributed/ComponentClassDescriptionImpl.h>
#include <string>

namespace SCIRun
{

  namespace Plume = sci::cca::plume;

  /** \class CCAComponentClassDescriptionImpl
   *
   */
  template<class Base>
  class CCAComponentClassDescriptionImpl : public ComponentClassDescriptionImpl<Base>
  {
  public:
    typedef Plume::CCAComponentClassDescription Interface;
    typedef Interface::pointer pointer;

    CCAComponentClassDescriptionImpl( const std::string &type, const std::string &library);
    virtual ~CCAComponentClassDescriptionImpl();

    virtual std::string getLibrary();

  private:
    std::string library;

    CCAComponentClassDescriptionImpl(const CCAComponentClassDescriptionImpl&);
    CCAComponentClassDescriptionImpl& operator=(const CCAComponentClassDescriptionImpl&);
  };
  
} // end namespace SCIRun

#endif
