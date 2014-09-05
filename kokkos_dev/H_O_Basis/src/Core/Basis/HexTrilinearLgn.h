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
//    File   : HexTrilinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(HexTrilinearLgn_h)
#define HexTrilinearLgn_h


#include <string>
#include <float.h>

#include <Core/Datatypes/TypeName.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Basis/Locate.h>


namespace SCIRun {

using std::string;

//! Class for creating geometrical approximations of Hex meshes
class HexApprox {  
public:
  //!< Parametric coordinates of vertices of unit edge
  static double UnitVertices[8][3];
  //!< References to vertices of unit edge 
  static int UnitEdges[12][2];
  //!< References to vertices of unit face 
  static int UnitFaces[6][4]; 

  HexApprox() {}
  virtual ~HexApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords gives parametric coordinates of the approximation.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			   vector<vector<double> > &coords) const
  {
    coords.resize(div_per_unit + 1);

    const double *v0 = UnitVertices[UnitEdges[edge][0]];
    const double *v1 = UnitVertices[UnitEdges[edge][1]];

    const double &p1x = v0[0];
    const double &p1y = v0[1];
    const double &p1z = v0[2];
    const double dx = v1[0] - p1x;
    const double dy = v1[1] - p1y;
    const double dz = v1[2] - p1z;

    for(unsigned i = 0; i <= div_per_unit; i++) {
      vector<double> &tmp = coords[i];
      tmp.resize(3);
      const double d = (double)i / (double)div_per_unit;
      tmp[0] = p1x + d * dx;
      tmp[1] = p1y + d * dy;
      tmp[2] = p1z + d * dz;
    } 	
  }
  
  //! return number of vertices per face 
  virtual int get_approx_face_elements() const { return 4; }
  
  //! Approximate faces for element by piecewise linear elements
  //! return: coords gives parametric coordinates at the approximation point.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_face(const unsigned face, 
			   const unsigned div_per_unit, 
			   vector<vector<vector<double> > > &coords) const
  {
    const double *v0 = UnitVertices[UnitFaces[face][0]];
    const double *v1 = UnitVertices[UnitFaces[face][1]];
    //	const double *v2=UnitVertices[UnitFaces[face][2]];
    const double *v3 = UnitVertices[UnitFaces[face][3]];
    const double d = 1. / (double)div_per_unit;
    coords.resize(div_per_unit);
    vector<vector<vector<double> > >::iterator citer = coords.begin();
    for(unsigned j = 0; j < div_per_unit; j++) {
      const double dj = (double)j / (double)div_per_unit;
      vector<vector<double> > &jvec = *citer++;
      jvec.resize((div_per_unit + 1) * 2, vector<double>(3, 0.0));
      vector<vector<double> >::iterator e = jvec.begin(); 
      for(unsigned i=0; i <= div_per_unit; i++) {
	const double di = (double) i / (double)div_per_unit;
	vector<double> &c0 = *e++;
	c0[0] = v0[0] + dj * (v3[0] - v0[0]) + di * (v1[0] - v0[0]);
	c0[1] = v0[1] + dj * (v3[1] - v0[1]) + di * (v1[1] - v0[1]);
	c0[2] = v0[2] + dj * (v3[2] - v0[2]) + di * (v1[2] - v0[2]);
	vector<double> &c1 = *e++;
	c1[0] = v0[0] + (dj + d) * (v3[0] - v0[0]) + di * (v1[0] - v0[0]);
	c1[1] = v0[1] + (dj + d) * (v3[1] - v0[1]) + di * (v1[1] - v0[1]);
	c1[2] = v0[2] + (dj + d) * (v3[2] - v0[2]) + di * (v1[2] - v0[2]);
      }
    }
  }
};

  
//! Class for searching of parametric coordinates related to a 
//! value in Hex meshes and fields
template <class ElemBasis>
class HexLocate : public Dim3Locate<ElemBasis> {
public:
  typedef typename ElemBasis::value_type T;

  HexLocate() {}
  virtual ~HexLocate() {}
 
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
    if (x[0]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	x[0]<=Dim3Locate<ElemBasis>::thresholdDist1)
      if (x[1]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	  x[1]<=Dim3Locate<ElemBasis>::thresholdDist1)
	if (x[2]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	    x[2]<=Dim3Locate<ElemBasis>::thresholdDist1) 
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
    vector<double> coord(3);
    guess.resize(3);
    for (int x = 0; x <= end; x++) {
      coord[0] = x / (double) end;
      for (int y = 0; y <= end; y++) {
	coord[1] = x / (double) end;
	for (int z = 0; z <= end; z++) {
	  coord[2] = x / (double) end;

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
  }
};



//! Class with weights and coordinates for 2nd order Gaussian integration
template <class T>
class HexGaussian2 
{
public:
  static int GaussianNum;
  static T GaussianPoints[8][3];
  static T GaussianWeights[8];
};

template <class T>
int HexGaussian2<T>::GaussianNum = 8;
  
template <class T>
T HexGaussian2<T>::GaussianPoints[8][3] = {
  {0.211324865405, 0.211324865405, 0.211324865405},
  {0.788675134595, 0.211324865405, 0.211324865405},
  {0.788675134595, 0.788675134595, 0.211324865405},
  {0.211324865405, 0.788675134595, 0.211324865405},
  {0.211324865405, 0.211324865405, 0.788675134595},
  {0.788675134595, 0.211324865405, 0.788675134595},
  {0.788675134595, 0.788675134595, 0.788675134595},
  {0.211324865405, 0.788675134595, 0.788675134595}};
  
template <class T>
T HexGaussian2<T>::GaussianWeights[8] = 
  {.125, .125, .125, .125, .125, .125, .125, .125};

//! Class with weights and coordinates for 3rd order Gaussian integration
template <class T>
class HexGaussian3 
{
public:
  static int GaussianNum;
  static T GaussianPoints[27][3];
  static T GaussianWeights[27];
};

template <class T>
T HexGaussian3<T>::GaussianPoints[27][3] = 
  {
    {0.11270166537950, 0.11270166537950, 0.11270166537950}, {0.5, 0.11270166537950, 0.11270166537950}, {0.88729833462050, 0.11270166537950, 0.11270166537950},
    {0.11270166537950, 0.5, 0.11270166537950}, {0.5, 0.5, 0.11270166537950}, {0.88729833462050, 0.5, 0.11270166537950},
    {0.11270166537950, 0.88729833462050, 0.11270166537950}, {0.5, 0.88729833462050, 0.11270166537950}, {0.88729833462050, 0.88729833462050, 0.11270166537950},
      
    {0.11270166537950, 0.11270166537950, 0.5}, {0.5, 0.11270166537950, 0.5}, {0.88729833462050, 0.11270166537950, 0.5},
    {0.11270166537950, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.88729833462050, 0.5, 0.5},
    {0.11270166537950, 0.88729833462050, 0.5}, {0.5, 0.88729833462050, 0.5}, {0.88729833462050, 0.88729833462050, 0.5},
      
    {0.11270166537950, 0.11270166537950, 0.88729833462050}, {0.5, 0.11270166537950, 0.88729833462050}, {0.88729833462050, 0.11270166537950, 0.88729833462050},
    {0.11270166537950, 0.5, 0.88729833462050}, {0.5, 0.5, 0.88729833462050}, {0.88729833462050, 0.5, 0.88729833462050},
    {0.11270166537950, 0.88729833462050, 0.88729833462050}, {0.5, 0.88729833462050, 0.88729833462050}, {0.88729833462050, 0.88729833462050, 0.88729833462050}
  };
  
template <class T>
T HexGaussian3<T>::GaussianWeights[27] = 
  {  
    0.03429355278944,   0.05486968447298,   0.03429355278944,
    0.05486968447298,   0.08779149517257,   0.05486968447298,
    0.03429355278944,   0.05486968447298,   0.03429355278944,

    0.03429355278944,   0.05486968447298,   0.03429355278944,
    0.05486968447298,   0.08779149517257,   0.05486968447298,
    0.03429355278944,   0.05486968447298,   0.03429355278944,

    0.03429355278944,   0.05486968447298,   0.03429355278944,
    0.05486968447298,   0.08779149517257,   0.05486968447298,
    0.03429355278944,   0.05486968447298,   0.03429355278944
  };

  
//! Class for handling of element of type hexahedron with trilinear lagrangian interpolation
template <class T>
class HexTrilinearLgn : public HexApprox, public HexGaussian2<double>
{
public:
  typedef T value_type;

  HexTrilinearLgn() {}
  virtual ~HexTrilinearLgn() {}
  
  int polynomial_order() const { return 1; }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];

    return (T)( -((-1 + x)*(-1 + y)*(-1 + z)) * cd.node0() 
		+x*(-1 + y)*(-1 + z) * cd.node1()
		-(x*y*(-1 + z)) * cd.node2()
		+(-1 + x)*y*(-1 + z) * cd.node3()
		+(-1 + x)*(-1 + y)*z * cd.node4() 
		-(x*(-1 + y)*z) * cd.node5() 
		+x*y*z * cd.node6() 
		-((-1 + x)*y*z) * cd.node7());
  }
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<T> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];

    derivs.resize(3);
    derivs[0] = T((-1 + y + z - y * z) * cd.node0()
		  + (-1 + y) * (-1 + z) * cd.node1()
		  + (y - y * z) * cd.node2()
		  + y * (-1 + z) * cd.node3()
		  + (-1 + y) * z * cd.node4()
		  + (z - y * z) * cd.node5()
		  + y * z * cd.node6()
		  - (y * z) * cd.node7());
      
    derivs[1] = T((-1 + x + z - x * z) * cd.node0()
		  + x * (-1 + z) * cd.node1()
		  + (x - x * z) * cd.node2()
		  + (-1 + x) * (-1 + z) * cd.node3()
		  + (-1 + x) * z * cd.node4()
		  - (x * z) * cd.node5()
		  + x * z * cd.node6()
		  + (z - x * z) * cd.node7());
      
    derivs[2] = T((-1 + x + y - x * y) * cd.node0()
		  + x * (-1 + y) * cd.node1()
		  - (x * y) * cd.node2()
		  + (-1 + x) * y * cd.node3()
		  + (-1 + x) * (-1 + y) * cd.node4()
		  + (x - x * y) * cd.node5()
		  + x * y * cd.node6()
		  + (y - x * y) * cd.node7());
  }
  
  //! get parametric coordinate for value within the element
  template <class CellData>
  bool get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
  {
    HexLocate< HexTrilinearLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  };
    
  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
};




template <class T>
const TypeDescription* 
get_type_description(HexTrilinearLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(HexTrilinearLgn<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}


template <class T>
const string
HexTrilinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("HexTrilinearLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}

const int HEX_TRILINEAR_LGN_VERSION = 1;
template <class T>
void
HexTrilinearLgn<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), HEX_TRILINEAR_LGN_VERSION);
  stream.end_class();
}
    
} //namespace SCIRun

#endif // HexTrilinear_h
