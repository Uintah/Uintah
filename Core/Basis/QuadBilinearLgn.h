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
//    File   : QuadBilinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(QuadBilinearLgn_h)
#define QuadBilinearLgn_h

#include <Core/Basis/Basis.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Basis/Locate.h>

namespace SCIRun {

using std::string;

//! Class for describing unit geometry of QuadBilinearLgn 
class QuadBilinearLgnUnitElement {
public: 
  //!< Parametric coordinates of vertices of unit edge
  static double unit_vertices[4][2];
  //!< References to vertices of unit edge  
  static int unit_edges[4][2]; 
  //!< References to vertices of unit face 
  static int unit_faces[1][4]; 
  
  QuadBilinearLgnUnitElement() {}
  virtual ~QuadBilinearLgnUnitElement() {}
  //! return dimension of domain 
  static int domain_dimension() { return 2; }
  //! return number of vertices
  static int number_of_vertices() { return 4; }
  //! return number of edges
  static int number_of_edges() { return 4; }
  //! return number of vertices per face 
  static int vertices_of_face() { return 4; }
  //! return number of faces per cell 
  static int faces_of_cell() { return 1; }
};


//! Class for creating geometrical approximations of Quad meshes
class QuadApprox {  
public:
  QuadApprox() {}
  virtual ~QuadApprox() {}
   
  //! Approximate edge for element by piecewise linear segments
  //! return: coords gives parametric coordinates of the approximation.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_edge(const unsigned edge, 
			   const unsigned div_per_unit, 
			   vector<vector<double> > &coords) const
  {
    coords.resize(div_per_unit + 1);
      
    const double p1x = QuadBilinearLgnUnitElement::unit_vertices[QuadBilinearLgnUnitElement::unit_edges[edge][0]][0];
    const double p1y = QuadBilinearLgnUnitElement::unit_vertices[QuadBilinearLgnUnitElement::unit_edges[edge][0]][1];
    const double dx = QuadBilinearLgnUnitElement::unit_vertices[QuadBilinearLgnUnitElement::unit_edges[edge][1]][0] - p1x;
    const double dy = QuadBilinearLgnUnitElement::unit_vertices[QuadBilinearLgnUnitElement::unit_edges[edge][1]][1] - p1y;
      
    for(unsigned i = 0; i <= div_per_unit; i ++) {
      const double d = (double)i / (double)div_per_unit;
      vector<double> &tmp = coords[i];
      tmp.resize(2);
      tmp[0] = p1x + d * dx;
      tmp[1] = p1y + d * dy;
    } 	      
  }
   
  //! Approximate faces for element by piecewise linear elements
  //! return: coords gives parametric coordinates at the approximation point.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_face(const unsigned /* face */, 
			   const unsigned div_per_unit, 
			   vector<vector<vector<double> > > &coords) const
  {
    coords.resize(div_per_unit);
    double d = 1. / div_per_unit;
    for(unsigned j = 0; j < div_per_unit; j++) {
      coords[j].resize(2 * (div_per_unit + 1));
      unsigned e = 0;
      for(unsigned i = 0; i <= div_per_unit; i++) {
	vector<double> &e1 = coords[j][e++];
	e1.resize(2);
	e1[0] = (double)i / (double)div_per_unit;
	e1[1] = (double)j / (double)div_per_unit;
	vector<double> &e2 = coords[j][e++];
	e2.resize(2);
	e2[0] = e1[0];
	e2[1] = (double)j / (double)div_per_unit + d;
      }
    }
  }
};
 
//! Class for searching of parametric coordinates related to a 
//! value in Quad meshes and fields
//! to do
template <class ElemBasis>
class QuadLocate : public Dim2Locate<ElemBasis> {
public:
  typedef typename ElemBasis::value_type T;

  QuadLocate() {}
  virtual ~QuadLocate() {}
 
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
    if (x[0]>=-Dim2Locate<ElemBasis>::thresholdDist && 
	x[0]<=Dim2Locate<ElemBasis>::thresholdDist1)
      if (x[1]>=-Dim2Locate<ElemBasis>::thresholdDist && 
	  x[1]<=Dim2Locate<ElemBasis>::thresholdDist1)
	return true;

    return false;
  }
  
  //! find a reasonable initial guess 
  template <class CellData>
  void initial_guess(const ElemBasis *pElem, const T &val, const CellData &cd, 
		     vector<double> & guess) const
  {
    double dist = DBL_MAX;
	
    vector<double> coord(2);
    vector<T> derivs(2);
    guess.resize(2);

    const int end = 2;
    for (int x = 0; x <= end; x++) {
      coord[0] = x / (double) end;
      for (int y = 0; y <= end; y++) {
	coord[1] = x / (double) end;
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
};


//! Class with weights and coordinates for 2nd order Gaussian integration
template <class T>
class QuadGaussian2 
{
public:
  static int GaussianNum;
  static T GaussianPoints[4][2];
  static T GaussianWeights[4];
};

template <class T>
int QuadGaussian2<T>::GaussianNum = 4;

template <class T>
T QuadGaussian2<T>::GaussianPoints[4][2] = {
  {0.211324865405, 0.211324865405},
  {0.788675134595, 0.211324865405},
  {0.788675134595, 0.788675134595},
  {0.211324865405, 0.788675134595}};

template <class T>
T QuadGaussian2<T>::GaussianWeights[4] = {.25, .25, .25, .25};

//! Class with weights and coordinates for 3rd order Gaussian integration
template <class T>
class QuadGaussian3 
{
public:
  static int GaussianNum;
  static T GaussianPoints[9][2];
  static T GaussianWeights[9];
};

template <class T>
int QuadGaussian3<T>::GaussianNum = 9;

template <class T>
T QuadGaussian3<T>::GaussianPoints[9][2] = {
  {0.11270166537950, 0.11270166537950}, {0.5, 0.11270166537950}, {0.88729833462050, 0.11270166537950},
  {0.11270166537950, 0.5}, {0.5, 0.5}, {0.88729833462050, 0.5},
  {0.11270166537950, 0.88729833462050}, {0.5, 0.88729833462050}, {0.88729833462050, 0.88729833462050}
};
  
template <class T>
T QuadGaussian3<T>::GaussianWeights[9] = {
  0.07716049378395,   0.12345679007654,   0.07716049378395,
  0.12345679007654,   0.19753086415802,   0.12345679007654,
  0.07716049378395,   0.12345679007654,   0.07716049378395
};

//! Class for handling of element of type quad with 
//! bilinear lagrangian interpolation
template <class T>
class QuadBilinearLgn : public BasisSimple<T>, 
                        public QuadApprox, 
			public QuadGaussian2<double>, 
			public QuadBilinearLgnUnitElement 
{
public:
  typedef T value_type;

  QuadBilinearLgn() {}
  virtual ~QuadBilinearLgn() {}
  
  int polynomial_order() const { return 1; }

  inline
  int get_weights(const vector<double> &coords, double *w) const
  { 
    const double x = coords[0], y = coords[1];  

    w[0] = (-1 + x) * (-1 + y);
    w[1] = -x * (-1 + y);
    w[2] = x * y;
    w[3] = -(-1 + x) * y;

    return 4;
  }

  //! get value at parametric coordinate
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    double w[4];
    get_weights(coords, w); 

    return (T)(w[0] * cd.node0() +
	       w[1] * cd.node1() +
	       w[2] * cd.node2() +
	       w[3] * cd.node3());
  }
    
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x = coords[0], y = coords[1];  

    derivs.resize(2);

    derivs[0] = T((-1 + y) * cd.node0()
		  +(1 - y) * cd.node1()
		  +y * cd.node2()
		  -y * cd.node3());

    derivs[1] = T((-1 + x) * cd.node0()
		  -x * cd.node1()
		  +x * cd.node2()
		  +(1 - x) * cd.node3());
  }  

  //! get parametric coordinate for value within the element
  template <class ElemData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const
  {
    QuadLocate< QuadBilinearLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }  
  
  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
};

template <class T>
const TypeDescription* get_type_description(QuadBilinearLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("QuadBilinearLgn", subs, 
				string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

template <class T>
const string
QuadBilinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("QuadBilinearLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}



const int QUADBILINEARLGN_VERSION = 1;
template <class T>
void
QuadBilinearLgn<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     QUADBILINEARLGN_VERSION);
  stream.end_class();
}

} //namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn back on 'implicit conversion... loss of accuracy' messages.
#  pragma reset woff 1506
#endif

#endif // QuadBilinearLgn_h
