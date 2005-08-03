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
 *  ProxyBase.h: Base class for all PIDL proxies
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_ProxyBase_h
#define CCA_PIDL_ProxyBase_h

#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/PIDL/ReferenceMgr.h>
#include <Core/CCA/PIDL/MxNScheduler.h>
#include <Core/CCA/PIDL/XceptionRelay.h>
#include <Core/CCA/Comm/PRMI.h>
#include <sgi_stl_warnings_off.h>
#include <vector> 
#include <string>
#include <sgi_stl_warnings_on.h>
#include <sstream>

namespace SCIRun {

    class XceptionRelay;

/**************************************
 
CLASS
   ProxyBase
   
KEYWORDS
   Proxy, PIDL
   
DESCRIPTION
   The base class for all proxy objects.  It contains the reference to
   the remote object (or references to all objects part of a parallel
   component).  This class should not be used outside of PIDL
   or automatically generated sidl code.
****************************************/

  class ProxyBase {
  public:
    
    //////////
    // A MxN Distribution Scheduler in case this proxy needs
    // to redistribute an array
    MxNScheduler* d_sched;
    
  protected:
    
    ProxyBase();
    
    ////////////
    // Create the proxy from the given reference.
    //remove it later
    ProxyBase(const Reference&);
    
    ////////////
    // Create the proxy from the given reference pointer.
    // the pointer is deleted inside the constructor
    ProxyBase(Reference*);
    
    ////////////
    // Create the proxy from the given reference list.
    ProxyBase(const ReferenceMgr&);	  
    
    ///////////
    // Destructor which closes connection
    virtual ~ProxyBase();
    
    //////////
    // Reference manager
    ReferenceMgr rm; 
    
    //////////
    // Xception relay which provides implementation of imprecise exceptions
    XceptionRelay* xr;
    
    //////////
    // TypeInfo is a friend so that it can call _proxyGetReference
    friend class TypeInfo;
    /////////
    // Xception is a friend since it needs access to the reference mgr's intracomm
    friend class XceptionRelay;
    /////////
    // Object_proxy is a friend since it uses proxyGetReferenceMgr()
    friend class Object_proxy;

    //////////
    // Return a the first internal reference or a copy of it.  
    void _proxyGetReference(Reference& ref, bool copy) const;
    
    //////////
    // Returns the reference manager 
    ReferenceMgr* _proxyGetReferenceMgr() const;
    
    //////////
    // Returns the unique identifier of this proxy 
    //    ::std::string getProxyUUID(); 
    ProxyID getProxyUUID();

    /////////
    // Created a subset of processes on the callee to service
    // all collective calls.
    void _proxycreateSubset(int localsize, int remotesize);

    /////////
    // Set rank and size of this component; and reset the values
    // to the defaults.
    void _proxysetRankAndSize(int rank, int size);
    void _proxyresetRankAndSize();

    /////////
    // Synchronizes parallel processes and receives all existing exceptions
    // Returns: exception id of prevailing exception 
    int _proxygetException(int xid=0, int lineid=0); 

  private:
    
    ////////
    // A unique name representing the entire proxy component
    // (Mostly used to represent the same proxy among the proxy's 
    // parallel processes) 
    //::std::string proxy_uuid; 
    ProxyID proxy_uuid;
  };
} // End namespace SCIRun

#endif





