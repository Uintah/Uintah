//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
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
//    File   : NoData.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mon Aug 23 09:57:14 2004

#if !defined(NoData_h)
#define NoData_h

#include <vector>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/TypeName.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off 'implicit conversion... loss of accuracy' messages.
#  pragma set woff 1506
#endif

namespace SCIRun {

using std::vector;
using std::string;

//! Class for handling of element without storage for field variables
template <class T> //! for compilation consistency
class NoDataBasis : public Persistent
{
public:
  NoDataBasis() {}
  virtual ~NoDataBasis() {}
  
  int polynomial_order() const { return -1; }
  //!< return dimension of domain 
  static int domain_dimension() { return -1; }
   //!< return number of vertices
  static int number_of_vertices() { return 0; }
  //!< return number of edges
  static int number_of_edges() { return 0; } 
  //!< return number of vertices per face 
  static int vertices_of_face() { return 0; } 
  //!< return number of faces per cell 
  static int faces_of_cell() { return 0; } 
  
  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    ASSERTFAIL("Data associated with basis 'NoDataBasis'");
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    ASSERTFAIL("Data associated with basis 'NoDataBasis'");
  }
  
  //! get parametric coordinate for value within the element
  //! iterative solution...
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    ASSERTFAIL("Coordinates cannot be cassociated with basis 'NoDataBasis'");
  }
  
  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
  
};

template <class T>
const string
NoDataBasis<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("NoDataBasis");
    return nm;
  }
  else 
  {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription* get_type_description(NoDataBasis<T> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription(NoDataBasis<T>::type_name(-1), 
				string(__FILE__), "SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

#define NODATABASIS_VERSION 1

template <class T>
void
NoDataBasis<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     NODATABASIS_VERSION);
  stream.end_class();
}

} // end namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn back on 'implicit conversion... loss of accuracy' messages.
#  pragma reset woff 1506
#endif

#endif // NoData_h
