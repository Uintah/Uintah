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
//    File   : PrismLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(PrismLinearLgn_h)
#define PrismLinearLgn_h

#include <Core/Basis/Basis.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Basis/Locate.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off 'implicit conversion... loss of accuracy' messages.
#  pragma set woff 1506
#endif

namespace SCIRun {

//! Class for describing unit geometry of PrismLinearLgn 
class PrismLinearLgnUnitElement {
public:
  static double unit_vertices[6][3]; //!< Parametric coordinates of vertices of unit edge
  static int unit_edges[9][3]; //!< References to vertices of unit edge
  static int unit_faces[5][4]; //!< References to vertices of unit face
 
  PrismLinearLgnUnitElement() {};
  virtual ~PrismLinearLgnUnitElement() {}
  
  static int domain_dimension() { return 3; } //! return dimension of domain 
  
  static int number_of_vertices() { return 6; } //! return number of vertices
  static int number_of_edges() { return 9; } //! return number of edges
  
  static int vertices_of_face() { return 3; } //! return number of vertices per face 

  static int faces_of_cell() { return 5; } //! return number of faces per cell 
};

//! Class for creating geometrical approximations of Prism meshes
class PrismApprox {  
public:

  PrismApprox() {}
  virtual ~PrismApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords gives parametric coordinates of the approximation.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			   vector<vector<double> > &coords) const
  {
    coords.resize(div_per_unit + 1);

    const double *v0 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_edges[edge][0]];
    const double *v1 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_edges[edge][1]];

    const double &p1x = v0[0];
    const double &p1y = v0[1];
    const double &p1z = v0[2];
    const double dx = v1[0] - p1x;
    const double dy = v1[1] - p1y;
    const double dz = v1[2] - p1z;

    for(unsigned i = 0; i <= div_per_unit; i++) {
      const double d = (double)div_per_unit / (double)i;
      vector<double> &tmp = coords[i];
      tmp[0] = p1x + d * dx;
      tmp[1] = p1y + d * dy;
      tmp[2] = p1z + d * dz;
    } 	
  }
    
  //! Approximate faces for element by piecewise linear elements
  //! return: coords gives parametric coordinates at the approximation point.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_face(const unsigned face, const unsigned div_per_unit, 
			   vector<vector<vector<double> > > &coords) const
  {	
    int fe = (PrismLinearLgnUnitElement::unit_faces[face][4] != -1 ? 2 : 1);
    coords.resize(fe * div_per_unit);
	
    for(int f = 0; f<2; f++) {
      double *v0, *v1, *v2;

      if (f==0) {
	v0 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][0]];
	v1 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][1]];
	v2 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][3]];
      } else {
	v0 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][2]];
	v1 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][3]];
	v2 = PrismLinearLgnUnitElement::unit_vertices[PrismLinearLgnUnitElement::unit_faces[face][1]];
      }

      const double d = 1. / div_per_unit;
      for(unsigned j = 0; j < div_per_unit; j++) {
	const double dj = (double)div_per_unit / (double)j;
	unsigned e = 0;
	coords[j + f * div_per_unit].resize((div_per_unit - j) * 2 + 1);
	vector<double> &tmp = coords[j + f * div_per_unit][e++];
	tmp[0] = v0[0] + dj * (v2[0] - v0[0]);
	tmp[1] = v0[1] + dj * (v2[1] - v0[1]);
	tmp[2] = v0[2] + dj * (v2[2] - v0[2]);
	for(unsigned i = 0; i < div_per_unit - j; i++) {
	  const double di = (double)div_per_unit / (double)i;
	  tmp = coords[j + f * div_per_unit][e++]; 
	  tmp[0] = v0[0] + (dj + d) * (v2[0] - v0[0]) + di * (v1[0] - v0[0]);
	  tmp[1] = v0[1] + (dj + d) * (v2[1] - v0[1]) + di * (v1[1] - v0[1]);
	  tmp[2] = v0[2] + (dj + d) * (v2[2] - v0[2]) + di * (v1[2] - v0[2]);
	  tmp = coords[j + f * div_per_unit][e++];
	  tmp[0] = v0[0] + dj * (v2[0] - v0[0]) + (di + d) * (v1[0] - v0[0]);
	  tmp[1] = v0[1] + dj * (v2[1] - v0[1]) + (di + d) * (v1[1] - v0[1]);
	  tmp[2] = v0[2] + dj * (v2[2] - v0[2]) + (di + d) * (v1[2] - v0[2]);
	}
      }
    }
  }
};

//! Class for searching of parametric coordinates related to a value in Prism meshes and fields
//! to do
template <class ElemBasis>
class PrismLocate : public Dim3Locate<ElemBasis> {
public:
  typedef typename ElemBasis::value_type T;
 
  PrismLocate() {}
  virtual ~PrismLocate() {}
 
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
    if (x[0]>=-Dim3Locate<ElemBasis>::thresholdDist)
      if (x[1]>=-Dim3Locate<ElemBasis>::thresholdDist)
	if (x[2]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	    x[2]<=Dim3Locate<ElemBasis>::thresholdDist1) 
	  if (x[0]+x[1]<=Dim3Locate<ElemBasis>::thresholdDist1)
	    return true;

    return false;
  };
  
  //! find a reasonable initial guess 
  template <class CellData>
  void initial_guess(const ElemBasis *pElem, const T &val, const CellData &cd, 
		     vector<double> & guess) const
  {
    double dist = DBL_MAX;
	
    vector<double> coord(3);
    vector<T> derivs(3);
    guess.resize(3);

    const int end = 3;
    for (int x = 1; x < end; x++) {
      coord[0] = x / (double) end;
      for (int y = 1; y < end; y++) {
	coord[1] = y / (double) end;
	if (coord[0]+coord[1]>Dim3Locate<ElemBasis>::thresholdDist1)
	  break;
	for (int z = 1; z < end; z++) {
	  coord[2] = z / (double) end;
	  double cur_d;
	  if (compare_distance(pElem->interpolate(coord, cd), 
			       val, cur_d, dist)) {
	    pElem->derivate(coord, cd, derivs);
	    if (!check_zero(derivs)) {
	      dist = cur_d;
	      guess = coord;
	    }
	  }
	}
      }
    }
  } 
};


//! Class with weights and coordinates for 2nd order Gaussian integration
template <class T>
class PrismGaussian2 
{
public:
  static int GaussianNum;
  static T GaussianPoints[6][3];
  static T GaussianWeights[6];
};

template <class T>
int PrismGaussian2<T>::GaussianNum = 6;

template <class T>
T PrismGaussian2<T>::GaussianPoints[6][3] = {
  {1./6.,1./6., 0.211324865405}, {2./3.,1./6., 0.211324865405}, 
  {1./6.,2./3., 0.211324865405}, {1./6.,1./6., 0.788675134595}, 
  {2./3.,1./6., 0.788675134595}, {1./6.,2./3., 0.788675134595}};

template <class T>
T PrismGaussian2<T>::GaussianWeights[6] = 
  {1./6., 1./6., 1./6., 1./6., 1./6., 1./6.};
  
//! Class with weights and coordinates for 3rd order Gaussian integration
//! to do

//! Class for handling of element of type prism with linear lagrangian interpolation
template <class T>
class PrismLinearLgn : public BasisSimple<T>, 
                       public PrismApprox, 
		       public PrismGaussian2<double>, 
		       public  PrismLinearLgnUnitElement
{
public:
  typedef T value_type;

  PrismLinearLgn() {}
  virtual ~PrismLinearLgn() {}

  int polynomial_order() const { return 1; }

  //! get weight factors at parametric coordinate 
  inline
  int get_weights(const vector<double> &coords, double *w) const
  { 
    const double x = coords[0], y = coords[1], z = coords[2];  
 
    w[0] = (-1 + x + y) * (-1 + z);
    w[1] = - (x * (-1 + z));
    w[2] = - (y * (-1 + z));
    w[3] = - ((-1 + x + y) * z);
    w[4] = +x * z;
    w[5] = +y * z;

    return 6;
  }
  
  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    double w[6];
    get_weights(coords, w); 
 
    return (T)(w[0] * cd.node0() +
	       w[1] * cd.node1() +
	       w[2] * cd.node2() +
	       w[3] * cd.node3() +
	       w[4] * cd.node4() +
	       w[5] * cd.node5());
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x = coords[0], y = coords[1], z = coords[2]; 

    derivs.resize(3);
 
    derivs[0] = T((-1 + z) * cd.node0()
		  +(1 - z) * cd.node1()
		  - z * cd.node3()
		  +z * cd.node4());
      
    derivs[1] = T((-1 + z) * cd.node0()
		  + (1 - z) * cd.node2()
		  - z * cd.node3()
		  + z * cd.node5());
      
    derivs[2] = T((-1 + x + y) * cd.node0()
		  - x * cd.node1()
		  - y * cd.node2()
		  + (1 - x - y) * cd.node3()
		  + x * cd.node4()
		  + y * cd.node5());
  }

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    PrismLocate< PrismLinearLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }  

  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* get_type_description(PrismLinearLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("PrismLinearLgn", subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const string
PrismLinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("PrismLinearLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int PRISMLINEARLGN_VERSION = 1;
template <class T>
void
PrismLinearLgn<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     PRISMLINEARLGN_VERSION);
  stream.end_class();
}

} //namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn back on 'implicit conversion... loss of accuracy' messages.
#  pragma reset woff 1506
#endif


#endif // PrismLinearLgn_h
