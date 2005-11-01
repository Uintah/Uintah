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
//    File   : Constant.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mon Aug 23 09:57:14 2004

#if !defined(Constant_h)
#define Constant_h

#include <Core/Basis/Basis.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/TypeName.h>

namespace SCIRun {

using std::string;

//! Class for describing unit geometry of ConstantBasis 
class ConstantBasisElement {
public: 
  static double unit_vertices[0][0]; //!< Parametric coordinates of vertices 
  static int unit_edges[0][0];    //!< References to vertices of unit edge 

  ConstantBasisElement() {}
  virtual ~ConstantBasisElement() {}

  //!< return dimension of domain 
  static int domain_dimension() { return 0; } 
  //!< return number of vertices
  static int number_of_vertices() { return 0; }
  //!< return number of edges 
  static int number_of_edges() { return 0; } 
  //!< return number of vertices per face 
  static int vertices_of_face() { return 0; } 
  //!< return number of faces per cell 
  static int faces_of_cell() { return 0; } 
};


//! Class for handling of element with constant field variables
template <class T>
  class ConstantBasis : public BasisSimple<T>, public ConstantBasisElement
{
public:
  ConstantBasis() {}
  virtual ~ConstantBasis() {}
  
  int polynomial_order() const { return 0; }
  //!< return dimension of domain 

  //! get value at parametric coordinate 
  template <class CellData>
  T interpolate(const vector<double> &coords, const CellData &cd) const
  {
    return cd.elem();
  }
  
  //! get first derivative at parametric coordinate
  template <class CellData>
  void derivate(const vector<double> &coords, const CellData &cd, 
		vector<T> &derivs) const
  {
    vector<double>::size_type s=coords.size();
    derivs.resize(s);
    for(vector<double>::size_type si=0; si<s; si++)
      derivs[si] = T(0);
  }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
};

template <class T>
const string
ConstantBasis<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("ConstantBasis");
    return nm;
  }
  else 
  {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription* get_type_description(ConstantBasis<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("ConstantBasis", subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

#define CONSTANTBASIS_VERSION 1

template <class T>
void
ConstantBasis<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     CONSTANTBASIS_VERSION);
  stream.end_class();
}

}
#endif // Constant_h
