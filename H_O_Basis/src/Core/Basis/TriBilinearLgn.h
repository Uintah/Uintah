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
//    File   : TriBilinearLgn.h
//    Author : Martin Cole
//    Date   : Fri Sep 10 09:40:12 2004

#if !defined(TriBilinearLgn_h)
#define TriBilinearLgn_h

#include <vector>
#include <string>
#include <Core/Basis/LinearLagrangian.h>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

using std::vector;
using std::string;

//! Triangular topology
template <class T>
class TriBilinearLgn : public LinearLagrangianBase<T> {
public:
  TriBilinearLgn() {}
  virtual ~TriBilinearLgn() {}

  //! Piecewise Linear approximation to within epsilon accuracy.
  //! return: approx has the values on the curve that satisfy epsilon.
  //! return: coords has the [0:1] coordinate corresponding to nodes in approx.
  template <class ElemData>
  void pl_approx(vector<T> &approx, vector<double> &coords, 
		 const ElemData &cd, double epsilon) const;

  // Value at coord
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const;
  
  //! First derivative at coord.
  template <class ElemData>
  T derivate(const vector<double> &coords, const ElemData &cd) const;

  //! Second derivative at coord.
  template <class ElemData>
  T derivate2(const vector<double> &coords, const ElemData &cd) const;

  //! return the parametric coordinates for value within the element.
  //! iterative solution...
  template <class ElemData>
  void get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const;  

  //! add a derivative value. (must correspond the the topo_values_)
  void add_derivative(const T &p) { derivs_.push_back(p); }

  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);

protected:

  //! Functions for interpolating value in elem.
  inline 
  double psi3(double xi) const {
    return 1.0 - psi1(xi) - psi2(xi);
  } 
  
  inline
  double psi_deriv_3(double xi) const {
    return -2.0L;
  }
   //! Functions for interpolating second derivative in elem
  inline
  double psi_deriv2_3(double xi) const {
    return 0.0L;
  }
  double distance(const T&, const T&) const;

};


template <class T>
const TypeDescription* get_type_description(BiLinearLagrangianBasis<T> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription(BiLinearLagrangianBasis<T>::type_name(-1), 
				string(__FILE__), "SCIRun");
  }
  return td;
}

template <class T>
const string
BiLinearLagrangianBasis<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("BiLinearLagrangianBasis");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

template <class T>
template <class ElemData>
T 
BiLinearLagrangianBasis<T>::interpolate(const vector<double> &coords, 
				      const ElemData &cd) const 
{

}

template <class T>
template <class ElemData>
T 
BiLinearLagrangianBasis<T>::derivate(const vector<double> &coords, 
				   const ElemData &cd) const 
{
}

template <class T>
template <class ElemData>
T 
BiLinearLagrangianBasis<T>::derivate2(const vector<double> &coords, 
				    const ElemData &cd) const 
{

}

template <class T>
template <class ElemData>
void 
BiLinearLagrangianBasis<T>::get_coords(vector<double> &coords, const T& value, 
				     const ElemData &cd) const
{

}

const int TRIBILINEARLGN_BASIS_VERSION = 1;
template <class T>
void
BiLinearLagrangianBasis<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TRIBILINEARLGN_BASIS_VERSION);

  stream.end_class();
}

} //namespace SCIRun

#endif // TriBilinearLgn_h
