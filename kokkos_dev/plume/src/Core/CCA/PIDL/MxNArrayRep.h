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
 *  MxNArrayRep.h 
 *
 *  Written by:
 *   Kostadin Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002 
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef CCA_PIDL_MxNArrayRep_h
#define CCA_PIDL_MxNArrayRep_h

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <algorithm>
#include <sgi_stl_warnings_on.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <Core/CCA/SSIDL/array.h>
#include <Core/CCA/PIDL/Reference.h>

/**************************************
				       
  CLASS
    MxNArrayRep, Index
   
  DESCRIPTION
    This class is a representation of an array domain
    using an Index per each dimension. An Index contains
    3 pieces of information: first element, last element,
    and stride. The first and the last element number
    are considered inclusive in the description of the
    array. The MxNArrayRep class is used to represent
    both subdomains and the the global domain of a
    particular array. 

****************************************/

namespace SCIRun {

  ////////////
  // Enumerated type denoting caller and callee. Used to
  // tag an object as caller or callee in respect
  // to the distribution.
  typedef enum { caller, callee } sched_t;

  ////////////
  // Solves ax + by = gcd(a,b) for a given a
  // and b integer values. In addition, gcd(a,b) is 
  // returned by this function.
  int gcd(int a, int b, int &x, int &y);
  /////////////
  // Greatest Common Divisor of two integers 
  int gcd(int m,int n);
  ////////////
  // Least Common Multiplier of two integers
  int lcm(int m,int n);
  ///////////
  // Intesects two array slice descriptions
  int intersectSlice(int f1, int s1, int f2, int s2);
  
  class Index;
  
  class MxNArrayRep {
  public:

    ////////////////
    // Constructor which takes the number of dimensions of the
    // array, an array of Index (size of the number of dimensions), 
    // and an optional Object reference if the object that associated
    // with this representation is not this object.
    // dimarr will NOT be freed. 
    MxNArrayRep(int dimno, Index* dimarr[], Reference* remote_ref = NULL);

    ////////////////
    // Constructor which takes a two-dimensional array consisted of 
    // first,last, and stride in all of the dimensions. 
    // Also the constructor accepts an optional Object reference 
    // if the object that associated with this representation is not this object.
    MxNArrayRep(SSIDL::array2<int>& arr, Reference* remote_ref = NULL);

    ///////////////
    // Destructor
    virtual ~MxNArrayRep();

    //////////////
    // Creates and retrieves a two-dimensional array consisted of 
    // first,last, and stride in all of the dimensions. 
    SSIDL::array2<int> getArray();

    /////////////
    // Retrieves the number of dimensions
    unsigned int getDimNum();

    /////////////
    // Retrieves the number specified as first in the Index representing
    // the dimension dimno. Note that dimensions are expected to begin with
    // 1 and not 0 in this implementation.
    unsigned int getFirst(int dimno);

    /////////////
    // Retrieves the number specified as last in the index representing
    // the dimension dimno. Note that dimensions are expected to begin with
    // 1 and not 0 in this implementation.
    unsigned int getLast(int dimno);

    /////////////
    // Retrieves the number specified as stride in the index representing
    // the dimension dimno. Note that dimensions are expected to begin with
    // 1 and not 0 in this implementation. 
    unsigned int getStride(int dimno);

    /////////////
    // Retrieves the number specified as local stride in the index representing
    // the dimension dimno. Note that dimensions are expected to begin with
    // 1 and not 0 in this implementation. 
    unsigned int getLocalStride(int dimno);

    ////////////
    // Calculates the number of elements according to the first, last and
    // stride for a given dimension (first & last inclusive).
    // Note that dimensions are expected to begin with 1 and not 0 in
    // this implementation.
    unsigned int getSize(int dimno);
    
    
    ///////////
    // Determines wether two array descriptions have an intersection. 
    // It calls Intersect() in order to make that determination.
    bool isIntersect(MxNArrayRep* arep);

    //////////
    // Intersects one dimension of a given representation with this one. It
    // return an Index describing the intersection
    Index* Intersect(MxNArrayRep* arep, int dimno);

    /////////
    // Calls Intersect (above) for each dimension a compiler a MxNArrayRep
    // of Index of the intersections.
    MxNArrayRep* Intersect(MxNArrayRep* arep);

    ///////////
    // Returns stored reference pointer. The reference pointer is used to
    // associate this representation to a remote object.
    Reference* getReference();

    ///////////
    // Accessor and Mutator methods for the rank. The rank is used to represents
    // by way of an ordered integer which remote object this class belongs to.
    void setRank(int rank);
    int getRank();

    ///////////
    // Prints this class to stdout.
    void print(std::ostream& dbg);
    
    ///////////
    // Used to signify (used only in the callee case) weather the data associated
    // with this distribution has been recieved.
    bool received;

  private:

    ///////////
    // Array of Index representing first, last and stride for each dimension
    std::vector<Index>mydimarr;

    //////////
    // Number of dimensions
    int mydimno;

    /////////
    // Reference to a remote object (used only in caller case)
    Reference* remoteRef;   

    ////////
    // Rank of the object this representation belongs to
    int d_rank;
  };
  


  class Index {
  public:

    //////////////
    //Index constructor. It requires that first <= last. If this is not
    //the case it transposes first and last.
    //@param localStride -- this stride represents data packing in the local array.
    //		            If the data is not packed we want to make special 
    //		            considerations (see getLocalStride()).	
    Index(unsigned int first, unsigned int last, unsigned int stride, int localStride = 1);

    ///////////
    // Prints this class to stdout.
    void print(std::ostream& dbg);
    
    //////////
    // Data 
    unsigned int myfirst;
    unsigned int mylast;
    unsigned int mystride;
    int localStride;
  };      
  
  
} // End namespace SCIRun

#endif














