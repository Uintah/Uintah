/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
//  
//    File   : HexTrilinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(HexTrilinearLgn_h)
#define HexTrilinearLgn_h

#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>

#include <Core/Basis/share.h>

namespace SCIRun {

//! Class for describing unit geometry of HexTrilinearLgn 
class SCISHARE HexTrilinearLgnUnitElement {
public:
  //! Parametric coordinates of vertices of unit edge 
  static double unit_vertices[8][3];
  //! References to vertices of unit edge   
  static int unit_edges[12][2]; 
  //! References to vertices of unit face  
  static int unit_faces[6][4];  
  //! References to normals of unit face 
  static double unit_face_normals[6][3];  
 
  HexTrilinearLgnUnitElement() {}
  virtual ~HexTrilinearLgnUnitElement() {}

  //! return dimension of domain 
  static int domain_dimension() { return 3; } 
  //! return number of vertices
  static int number_of_vertices() { return 8; } 
  //! return number of vertices
  static int number_of_mesh_vertices() { return 8; } 
  //! return degrees of freedom
  static int dofs() { return 8; } 
  //! return number of edges
  static int number_of_edges() { return 12; } 
  //! return number of vertices per face 
  static int vertices_of_face() { return 4; } 
  //! return number of faces per cell 
  static int faces_of_cell() { return 6; } 

  static inline double length(int edge) {
    const double *v0 = unit_vertices[unit_edges[edge][0]];
    const double *v1 = unit_vertices[unit_edges[edge][1]];
    const double dx = v1[0] - v0[0];
    const double dy = v1[1] - v0[1];
    const double dz = v1[2] - v0[2];
    return sqrt(dx*dx+dy*dy+dz*dz);
  } 
  static double area(int /*face*/) { return 1.; }
  static double volume() { return 1.; }
};


//! Class for creating geometrical approximations of Hex meshes
class HexApprox {  
public:
 
  HexApprox() {}
  virtual ~HexApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords gives parametric coordinates of the approximation.
  //! Use interpolate with coordinates to get the world coordinates.
  virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			   std::vector<std::vector<double> > &coords) const
  {
    coords.resize(div_per_unit + 1);

    const double *v0 = HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_edges[edge][0]];
    const double *v1 = HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_edges[edge][1]];

    const double &p1x = v0[0];
    const double &p1y = v0[1];
    const double &p1z = v0[2];
    const double dx = v1[0] - p1x;
    const double dy = v1[1] - p1y;
    const double dz = v1[2] - p1z;

    for(unsigned i = 0; i <= div_per_unit; i++) {
      std::vector<double> &tmp = coords[i];
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
			   std::vector<std::vector<std::vector<double> > > &coords) const
  {
    const double *v0 = HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_faces[face][0]];
    const double *v1 = HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_faces[face][1]];
    //	const double *v2=HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_faces[face][2]];
    const double *v3 = HexTrilinearLgnUnitElement::unit_vertices[HexTrilinearLgnUnitElement::unit_faces[face][3]];
    const double d = 1. / (double)div_per_unit;
    coords.resize(div_per_unit);
    std::vector<std::vector<std::vector<double> > >::iterator citer = coords.begin();
    for(unsigned j = 0; j < div_per_unit; j++) {
      const double dj = (double)j / (double)div_per_unit;
      std::vector<std::vector<double> > &jvec = *citer++;
      jvec.resize((div_per_unit + 1) * 2, std::vector<double>(3, 0.0));
      std::vector<std::vector<double> >::iterator e = jvec.begin(); 
      for(unsigned i=0; i <= div_per_unit; i++) {
        const double di = (double) i / (double)div_per_unit;
        std::vector<double> &c0 = *e++;
        c0[0] = v0[0] + dj * (v3[0] - v0[0]) + di * (v1[0] - v0[0]);
        c0[1] = v0[1] + dj * (v3[1] - v0[1]) + di * (v1[1] - v0[1]);
        c0[2] = v0[2] + dj * (v3[2] - v0[2]) + di * (v1[2] - v0[2]);
        std::vector<double> &c1 = *e++;
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
  template <class ElemData>
  bool get_coords(const ElemBasis *pEB, std::vector<double> &coords, 
		  const T& value, const ElemData &cd) const  
  {      
    initial_guess(pEB, value, cd, coords);
    if (get_iterative(pEB, coords, value, cd))
      return check_coords(coords);
    return false;
  }

  inline bool check_coords(const std::vector<double> &x) const  
  {  
    if (x[0]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	x[0]<=Dim3Locate<ElemBasis>::thresholdDist1)
      if (x[1]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	  x[1]<=Dim3Locate<ElemBasis>::thresholdDist1)
	if (x[2]>=-Dim3Locate<ElemBasis>::thresholdDist && 
	    x[2]<=Dim3Locate<ElemBasis>::thresholdDist1) 
	  return true;

    return false;
  }
  
protected:
  //! find a reasonable initial guess 
  template <class ElemData>
  void initial_guess(const ElemBasis *pElem, const T &val, const ElemData &cd, 
		     std::vector<double> & guess) const
  {
    double dist = DBL_MAX;
	
    std::vector<double> coord(3);
    std::vector<T> derivs(3);
    guess.resize(3);

    const int end = 3;
    for (int x = 1; x < end; x++) {
      coord[0] = x / (double) end;
      for (int y = 1; y < end; y++) {
	coord[1] = y / (double) end;
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


//! Class with weights and coordinates for 1st order Gaussian integration
template <class T>
class HexGaussian1
{
public:
  static int GaussianNum;
  static T GaussianPoints[1][3];
  static T GaussianWeights[1];
};

#ifdef _WIN32
// force the instantiation of TetGaussian2<double>
template class HexGaussian1<double>;
#endif

template <class T>
int HexGaussian1<T>::GaussianNum = 1;
  
template <class T>
T HexGaussian1<T>::GaussianPoints[1][3] = {
  {0.5, 0.5, 0.5}};
  
template <class T>
T HexGaussian1<T>::GaussianWeights[1] = 
  {1.0};
 
//! Class with weights and coordinates for 2nd order Gaussian integration
template <class T>
class HexGaussian2 
{
public:
  static int GaussianNum;
  static T GaussianPoints[8][3];
  static T GaussianWeights[8];
};

#ifdef _WIN32
// force the instantiation of TetGaussian2<double>
template class HexGaussian2<double>;
#endif

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
int HexGaussian3<T>::GaussianNum = 27;
 
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

  
//! Class for handling of element of type hexahedron with 
//! trilinear lagrangian interpolation
template <class T>
class HexTrilinearLgn : public BasisSimple<T>, 
                        public HexApprox, 
			public HexGaussian2<double>, 
			public HexTrilinearLgnUnitElement
{
public:
  typedef T value_type;

  HexTrilinearLgn() {}
  virtual ~HexTrilinearLgn() {}
  
  static int polynomial_order() { return 1; }

  //! get weight factors at parametric coordinate 
  inline
  static void get_weights(const std::vector<double> &coords, double *w) 
  {
    const double x=coords[0], y=coords[1], z=coords[2];
    w[0] = -((-1 + x)*(-1 + y)*(-1 + z));
    w[1] = x*(-1 + y)*(-1 + z);
    w[2] = -(x*y*(-1 + z));
    w[3] = (-1 + x)*y*(-1 + z);
    w[4] = (-1 + x)*(-1 + y)*z;
    w[5] = -(x*(-1 + y)*z);
    w[6] = x*y*z;
    w[7] = -((-1 + x)*y*z);
  }

  //! get value at parametric coordinate 
  template <class ElemData>
  T interpolate(const std::vector<double> &coords, const ElemData &cd) const
  {
    double w[8];
    get_weights(coords, w); 
    return (T)( w[0] * cd.node0() +
		w[1] * cd.node1() +
		w[2] * cd.node2() +
		w[3] * cd.node3() +
		w[4] * cd.node4() +
		w[5] * cd.node5() +
		w[6] * cd.node6() +
		w[7] * cd.node7());
  }
  
  //! get derivative weight factors at parametric coordinate 
  inline
  static void get_derivate_weights(const std::vector<double> &coords, double *w) 
  {
   const double x=coords[0], y=coords[1], z=coords[2];  
    w[0]=(-1 + y + z - y * z);
    w[1]= (-1 + y) * (-1 + z);
    w[2]= (y - y * z);
    w[3]= y * (-1 + z);
    w[4]= (-1 + y) * z;
    w[5]= (z - y * z);
    w[6]=  y * z;
    w[7]=  -(y * z);
    w[8]=(-1 + x + z - x * z);
    w[9]= x * (-1 + z);
    w[10]=  (x - x * z);
    w[11]=  (-1 + x) * (-1 + z);
    w[12]=  (-1 + x) * z;
    w[13]=  -(x * z);
    w[14]=  x * z;
    w[15]=  (z - x * z);   
    w[16]=(-1 + x + y - x * y);
    w[17]= x * (-1 + y);
    w[18]=  -(x * y);
    w[19]= (-1 + x) * y;
    w[20]= (-1 + x) * (-1 + y);
    w[21]=  (x - x * y);
    w[22]=  x * y;
    w[23]=  (y - x * y);    
  }

  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const std::vector<double> &coords, const ElemData &cd, 
		std::vector<T> &derivs) const
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
  template <class ElemData>
  bool get_coords(std::vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
  {
    HexLocate< HexTrilinearLgn<T> > CL;
    return CL.get_coords(this, coords, value, cd);
  }
    
  //! get arc length for edge
  template <class ElemData>
  double get_arc_length(const unsigned edge, const ElemData &cd) const  
  {
    return get_arc3d_length<CrvGaussian1<double> >(this, edge, cd);
  }
 
  //! get area
  template <class ElemData>
    double get_area(const unsigned face, const ElemData &cd) const  
  {
    return get_area3<QuadGaussian3<double> >(this, face, cd);
  }
  
  //! get volume
  template <class ElemData>
    double get_volume(const ElemData & cd) const  
  {
    return get_volume3(this, cd);
  }
  
  static  const std::string type_name(int n = -1);

  virtual void io (Piostream& str);
};


template <class T>
const TypeDescription* 
get_type_description(HexTrilinearLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("HexTrilinearLgn", subs, 
				std::string(__FILE__),
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}


template <class T>
const std::string
HexTrilinearLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("HexTrilinearLgn");
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
  stream.begin_class(get_type_description(this)->get_name(),
                     HEX_TRILINEAR_LGN_VERSION);
  stream.end_class();
}
    
} //namespace SCIRun


#endif // HexTrilinear_h
