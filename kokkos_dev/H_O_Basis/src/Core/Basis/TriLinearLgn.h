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
//    File   : TriLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(TriLinearLgn_h)
#define TriLinearLgn_h

#include <vector>
#include <float.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Basis/Locate.h>

namespace SCIRun {

using std::vector;
using std::string;
  
//! Class for creating geometrical approximations of Tri meshes
class TriApprox {  
public:
  //!< Parametric coordinates of vertices of unit edge
  static double UnitVertices[3][2];
  //!< References to vertices of unit edge 
  static int UnitEdges[3][2]; 
  //!< References to vertices of unit face
  static int UnitFaces[1][3]; 

  TriApprox() {}
  virtual ~TriApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords gives parametric coordinates of the approximation.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			   vector<vector<double> > &coords) const
  {
    coords.resize(div_per_unit+1);

    const double *v0 = UnitVertices[UnitEdges[edge][0]];
    const double *v1 = UnitVertices[UnitEdges[edge][1]];

    const double &p1x = v0[0];
    const double &p1y = v0[1];
    const double dx = v1[0] - p1x;
    const double dy = v1[1] - p1y;

    for(unsigned i = 0; i <= div_per_unit; i++) {
      const double d = (double)div_per_unit / (double)i;
      vector<double> &tmp = coords[i];
      tmp[0] = p1x + d * dx;
      tmp[1] = p1y + d * dy;
    } 	
  } 
  
  //! return number of vertices per face 
  virtual int get_approx_face_elements() const { return 3; }
  
  //! Approximate faces for element by piecewise linear elements
  //! return: coords gives parametric coordinates at the approximation point.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_face(const unsigned /* face */, 
			   const unsigned div_per_unit, 
			   vector<vector<vector<double> > > &coords) const
  {
    coords.resize(div_per_unit);
    const double d = 1. / div_per_unit;

    for(unsigned j = 0; j < div_per_unit; j++) {
      const double dj = (double)div_per_unit / (double)j;
      unsigned e = 0;
      coords[j].resize((div_per_unit - j) * 2 + 1);
      vector<double> &tmp = coords[j][e++];
      tmp[0] = 0;
      tmp[1] = dj;
      for(unsigned i = 0; i<div_per_unit - j; i++) {
	const double di = (double)div_per_unit / (double)i;
	tmp = coords[j][e++];
	tmp[0] = di;
	tmp[1] = dj + d;
	tmp = coords[j][e++];
	tmp[0] = di + d;
	tmp[1] = dj;
      }
    }
  }
};

//! Class for searching of parametric coordinates related to a 
//! value in Tri meshes and fields
//! to do
template <class ElemBasis>
class TriLocate {
public:
  typedef typename ElemBasis::value_type T;

  TriLocate() {}
  virtual ~TriLocate() {}
 
  //! find value in interpolation for given value
  template <class CellData>
  bool get_coords(const ElemBasis *pEB, vector<double> &coords, 
		  const T& value, const CellData &cd) const  
  {          
    initial_guess(pEB, value, cd, coords);
    if (get_iterative(pEB, coords, value, cd))
      return check_coords(coords);
    return false;
  }

protected:
  inline bool check_coords(const vector<double> &x) const  
  {  
    if (x[0]>=-Dim2Locate<ElemBasis>::thresholdDist)
      if (x[1]>=-Dim2Locate<ElemBasis>::thresholdDist)
	if (x[0]+x[1]<=Dim2Locate<ElemBasis>::thresholdDist1)
	  return true;

    return false;
  };
  
  //! find a reasonable initial guess 
  template <class CellData>
  void initial_guess(const ElemBasis *pElem, const T &val, const CellData &cd, 
		     vector<double> & guess) const
  {
    double dist = DBL_MAX;
	
    int end = 4;
    vector<double> coord(2);
    guess.resize(2);
    for (int x = 0; x <= end; x++) {
      coord[0] = x / (double) end;
      for (int y = 0; y <= end; y++) {
	coord[1] = x / (double) end;
	if (coord[0]+coord[1]>Dim2Locate<ElemBasis>::thresholdDist1)
	  break;
	double cur_d;
	if (compare_distance(pElem->interpolate(coord, cd), 
			     val, cur_d, dist)) 
	{
	  dist = cur_d;
	  guess = coord;
	}
      }
    }
  }
};

//! Class with weights and coordinates for 2nd order Gaussian integration
template <class T>
class TriGaussian2 
{
public:
  static int GaussianNum;
  static T GaussianPoints[3][2];
  static T GaussianWeights[3];
};

template <class T>
int TriGaussian2<T>::GaussianNum = 3;

template <class T>
T TriGaussian2<T>::GaussianPoints[3][2] = {
  {1./6.,1./6.}, {2./3.,1./6.}, {1./6.,2./3.}};

template <class T>
T TriGaussian2<T>::GaussianWeights[3] = {1./3., 1./3., 1./3.};

//! Class with weights and coordinates for 3rd order Gaussian integration
template <class T>
class TriGaussian3 
{
public:
  static int GaussianNum;
  static T GaussianPoints[7][2];
  static T GaussianWeights[7];
};

template <class T>
int TriGaussian3<T>::GaussianNum = 7;

template <class T>
T TriGaussian3<T>::GaussianPoints[7][2] = {
  {0.1012865073, 0.1012865073}, {0.7974269853, 0.1012865073}, {0.1012865073, 0.7974269853},
  {0.4701420641, 0.0597158717}, {0.4701420641, 0.4701420641}, {0.0597158717, 0.4701420641},
  {0.3333333333, 0.3333333333}};

template <class T>
T TriGaussian3<T>::GaussianWeights[7] = 
  {0.1259391805, 0.1259391805, 0.1259391805, 0.1323941527, 0.1323941527, 0.1323941527, 0.0225};


//! Class for handling of element of type triangle with linear lagrangian interpolation
template <class T>
class TriLinearLgn : public TriApprox, public TriGaussian2<double>  
{ 
public:
  typedef T value_type;

  TriLinearLgn() {}
  virtual ~TriLinearLgn() {}
  
  int polynomial_order() const { return 1; }

  virtual void approx_edge(const unsigned edge, 
			   const unsigned /* div_per_unit */,
			   vector<vector<double> > &coords) const
  {
    coords.resize(2);
    vector<double> &tmp = coords[0];
    tmp[0] = UnitVertices[UnitEdges[edge][0]][0];
    tmp[1] = UnitVertices[UnitEdges[edge][0]][1];
    tmp = coords[1];
    tmp[0] = UnitVertices[UnitEdges[edge][1]][0];
    tmp[1] = UnitVertices[UnitEdges[edge][1]][1];
  }

  virtual void approx_face(const unsigned /* face */, 
			   const unsigned div_per_unit,
			   vector<vector<vector<double> > > &coords) const
  {
    coords.resize(1);
    coords[0].resize(3);
    vector<double> &tmp = coords[0][0];
    tmp[0] = 0;
    tmp[1] = 0;
    tmp = coords[0][1];
    tmp[0] = 1;
    tmp[1] = 0;
    tmp = coords[0][2];
    tmp[0] = 0;
    tmp[1] = 1;
  }

  //! get value at parametric coordinate
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    const double x = coords[0], y = coords[1];  
    return (T)((1 - x - y) * cd.node0() + x * cd.node1() + y * cd.node2());
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    derivs.resize(2);

    derivs[0] = T(-1. * cd.node0() + cd.node1());
    derivs[1] = T(-1. * cd.node0() + cd.node2());
  }
	
  //! get the parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    TriLocate< TriLinearLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  };
 
  static const string type_name(int n = -1);

  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* get_type_description(TriLinearLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(TriLinearLgn<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class T>
const string
TriLinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("TriLinearLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

const int TRILINEARLGN_VERSION = 1;
template <class T>
void 
TriLinearLgn<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), TRILINEARLGN_VERSION);
  stream.end_class();
}


} //namespace SCIRun

#endif // TriLinearLgn_h
