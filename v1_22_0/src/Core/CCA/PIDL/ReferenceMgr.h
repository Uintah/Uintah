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
 *  ReferenceMgr.h: Class which manages references to remote object. It is used
 *                    to enable parallel references and enable subset creation.
 *                      
 *                       .
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef CCA_PIDL_ReferenceMgr_h
#define CCA_PIDL_ReferenceMgr_h

#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/Comm/Intra/IntraComm.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   ReferenceMgr
   
KEYWORDS
   PIDL, Client, Proxy, Reference
   
DESCRIPTION
   The ReferenceMgr class is associated with a proxy object
   and it's purpose is to make manage the references to the
   remote component, regardless of whether this component is 
   parallel or serial. It also provides an interface to 
   facilitate the creation of subsets.

****************************************/
  class Object_proxy;
  class TypeInfo;

  typedef std::vector<Reference*> refList;

  typedef enum {
    REDIS = 1,
    CALLONLY,
    CALLNORET,
    NOCALLRET
  } callType;
    

  class ReferenceMgr {
  public:
    /////////
    // Default constructor
    ReferenceMgr();

    /////////
    // Constructor with rank/size (parallel proxies)
    ReferenceMgr(int rank, int size);

    //////////
    // Copy the referenc manager.
    ReferenceMgr(const ReferenceMgr&);


    //////////
    // Destructor which clears all current storage entries
    ~ReferenceMgr(); 

    //////////
    // Retreives a pointer to the reference needed for  
    // independent (if parallel) or serial RMI 
    Reference* getIndependentReference() const;

    //////////
    // Retreives a pointer to the reference needed for  
    // collective RMI
    ::std::vector<Reference*> getCollectiveReference(callType tip);

    /////////
    // Retreives a pointer to the list of all references. If the
    // current state of this object is contaminated (rank >= size)
    // we return an empty reference list. 
    refList* getAllReferences() const;

    //////////
    // Adds another reference (parallel component situation)
    void insertReference(Reference *ref);

    ///////////
    // Returns the number of references in the list
    int getRemoteSize();

    ///////////
    // Returns the number of references in the list
    int getSize();

    ///////////
    // Returns the number of references in the list
    int getRank();

    //////////
    // Create a subset. Affects all get*Reference() methods and 
    // the getRank(), getSize(), and getRemoteSize() methods. 
    // Passing 0 or a negative number resets the state
    // of this object to no subsets. Passing a
    // number larger than the  parallel size has no effect.
    void createSubset(int localsize, int remotesize);

    ////////
    // Modifies the size and the rank. Shouldn't be called together
    // with createSubset. The previous rank and size are not saved.
    void ReferenceMgr::setRankAndSize(int rank, int size);
    ///////
    // Resets the rank and size to that that of the default constuctor.
    void ReferenceMgr::resetRankAndSize();

    /////////
    // An object used to facilitate intra-component communication
    // (parallel component case)
    IntraComm* intracomm;
    
  protected:
    ///////
    // These class is involved in setting up ReferenceMgr
    friend class Object_proxy;
    friend class TypeInfo;
    friend class ProxyBase;

  private:
    /////////
    // For parallel proxies, number of cohorts
    int localSize;

    /////////
    // For parallel proxies, my ordered number
    int localRank;

    ////////
    // Subset specific value of localSize. Some methods return this
    // value in order to facilitate subsetting
    int s_lSize;

    ///////
    // Subset specific value of d_ref.size(). 
    int s_refSize;

    //////////
    // A vector of reference pointers to the remote objects.
    refList d_ref;
  };
} // End namespace SCIRun

#endif




