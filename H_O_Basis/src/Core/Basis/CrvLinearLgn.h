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
//    File   : CrvLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 03 2004

#if !defined(CrvLinearLgn_h)
#define CrvLinearLgn_h

#include <vector>
#include <string>
#include <Core/Basis/LinearLagrangian.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

using std::vector;
using std::string;

class CrvApprox {
public:
  static double UnitVertices[1][2];
  static int UnitEdges[1][2];

  CrvApprox() {}
  virtual ~CrvApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords is vector of the parametric coordinates at the 
  //!         approximation point. Use interpolate to get the values.
  virtual void approx_edge(const unsigned /* edge */, 
			   const unsigned div_per_unit, 
			   vector<vector<double> > &coords) const
  {
    coords.resize(div_per_unit + 1);
    for(unsigned i = 0; i <= div_per_unit; i++) {
      vector<double> &tmp = coords[i];
      tmp.resize(1);
      tmp[0] = (double)div_per_unit / (double)i;
    }
  }
  
  virtual int get_approx_face_elements() const { return 0; }
  
  virtual void approx_face(const unsigned /* face */, 
			   const unsigned /* div_per_unit */, 
			   vector<vector<double> > & /* coords */) const
  {}
};
 

//! Curve topology
template <class T>
class CrvLinearLgn : public CrvApprox
{
public:
  static int GaussianNum;
  static double GaussianPoints[1][1];
  static double GaussianWeights[1];

  CrvLinearLgn() : CrvApprox() {}
  virtual ~CrvLinearLgn() {}
  
  virtual void approx_edge(const unsigned /* edge */, 
			   const unsigned /*div_per_unit*/,
			   vector<vector<double> > &coords) const
  {
    coords.resize(2);
    vector<double> &tmp = coords[0];
    tmp.resize(1);
    tmp[0] = 0.0;
    tmp = coords[1];
    tmp.resize(1);
    tmp[0] = 1.0;
  }

  // Value at coord
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    const double x=coords[0];  
    return (T)((1-x)*cd.node0()+x*cd.node1());
  }
    
  //! First derivative at coord.
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<double> &derivs) const
  {
    derivs.size(1);

    derivs[0] = -cd.node0() + cd.node1();
  }

  //! return the parametric coordinates for value within the element.
  //! iterative solution...
  template <class ElemData>
  void get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const;  

  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);
protected:
  //! next_guess is the next Newton iteration step.
  template <class ElemData>
  double next_guess(double xi, const T &val, const ElemData &cd) const;

  //! find a reasonable initial guess for starting Newton iteration.
  template <class ElemData>
  double initial_guess(const T &val, const ElemData &cd) const;
};

template <class T>
const TypeDescription* get_type_description(CrvLinearLgn<T> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription(CrvLinearLgn<T>::type_name(-1), 
				string(__FILE__), "SCIRun");
  }
  return td;
}

template <class T>
const string
CrvLinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("CrvLinearLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

const int CRVLINEARLGN_VERSION = 1;
template <class T>
void
CrvLinearLgn<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), CRVLINEARLGN_VERSION);
  stream.end_class();
}

template <>
template <class ElemData>
void 
CrvLinearLgn<Point>::get_coords(vector<double> &coords, const Point& value, 
				const ElemData &cd) const;

template <class T>
template <class ElemData>
void 
CrvLinearLgn<T>::get_coords(vector<double> &coords, const T& value, 
			    const ElemData &cd) const
{
  double denominator = cd.node1() - cd.node0();
  double numerator   = value - cd.node0();
  coords[0] = numerator / denominator;
}
 

} //namespace SCIRun

#endif // LinearLagrangian_h
