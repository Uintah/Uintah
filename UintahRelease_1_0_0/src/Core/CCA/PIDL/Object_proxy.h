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
 *  Object_proxy.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_Object_proxy_h
#define CCA_PIDL_Object_proxy_h

#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/PIDL/Object.h>

namespace SCIRun {
/**************************************
 
CLASS
   Object_proxy
   
KEYWORDS
   Proxy, Object, PIDL
   
DESCRIPTION
   Internal PIDL class for a proxy to a base object.  This impements
   the Object interface and provides a proxy mechanism for
   remote objects.  Since there are no interesting methods at this level,
   the only interesting thing that we can do is up-cast.
****************************************/
  class Object_proxy : public ProxyBase, public Object {
  public:
  protected:
    //////////
    // PIDL will create these.
    friend class PIDL;

    //////////
    // Private constructor from a reference
    Object_proxy(const Reference&);

    //////////
    // Private constructor from a URL
    Object_proxy(const URL&);

    //////////
    // Private constructor from an array of URLs
    // (parallel component case)
    Object_proxy(const int urlc, const URL urlv[], int mysize, int myrank);

    //////////
    // Private constructor from an vector of URLs
    // (parallel component case)
    Object_proxy(const std::vector<URL>& urlv, int mysize, int myrank);

    //////////
    // Private constructor from an vector of proxies. Combine them all into one parallel component. 
    // (parallel component case)
    Object_proxy(const std::vector<Object::pointer>& pxy, int mysize, int myrank);

    //////////
    // Destructor
    virtual ~Object_proxy();

  private:
  };
} // End namespace SCIRun

#endif

