//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  he MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Basis.h
//    Author : Frank B. Sachse
//    Date   : Mon Oct 21 

#if !defined(Basis_h)
#define Basis_h

#include <vector>
#include <Core/Util/Assert.h>

namespace SCIRun {

using std::vector;


//! Class for handling of element with constant field variables
template <class T>
class BasisSimple 
{
public:
  BasisSimple() {}
  virtual ~BasisSimple() {}
  
  int polynomial_order() const { return 0; }

  //! get value at parametric coordinate 
  template <class CellData>
  T interpolate(const vector<double> &coords, const CellData &cd) const
  {
    ASSERTFAIL("interpolate not supported by basis");
  }
  
  //! get first derivative at parametric coordinate
  template <class CellData>
  void derivate(const vector<double> &coords, const CellData &cd, 
		vector<T> &derivs) const
  {
    ASSERTFAIL("derivate not supported by basis");
  }

  //! get parametric coordinate for value within the element
  template <class CellData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const
  {
    ASSERTFAIL("get_coords not supported by basis");
  }

  //! add a node value corresponding to edge
  void add_node_value(const T &p) 
  {
    ASSERTFAIL("add_node_value not supported by basis");
  }

  //! add a derivative value for nodes
  void add_derivative(const vector<T> &p) 
  {
    ASSERTFAIL("add_derivative not supported by basis");
  }
};


}
#endif // Basis_h
