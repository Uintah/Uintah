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

#include <vector>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

using std::vector;
using std::string;

class HexApprox {  
public:
  static double UnitVertices[8][3];
  static int UnitEdges[12][2];
  static int UnitFaces[6][4];

  HexApprox() {}
  virtual ~HexApprox() {}
  
  //! Approximate edge for element by piecewise linear segments
  //! return: coords has the parametric coordinates of the approximation.
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
  
  virtual int get_approx_face_elements() const { return 4; }
  
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

template <class T>
double val_type_length(const T& t);

template <>
double val_type_length(const Point& t);

template <class T>
double val_type_length(const T& t) {
  return t;
}

template <class T>
T val_type_difference(const T& t0, const T& t1);

template <>
Point val_type_difference(const Point& t0, const Point& t1);

template <class T>
T val_type_difference(const T& t0, const T& t1)
{
  return t0 - t1;
}
  
template <class HexBasis>
class HexLocate {
public:

  typedef typename HexBasis::value_type T;

  HexLocate() {}
  virtual ~HexLocate() {}
 
  //! iterative solution...
  template <class CellData>
  void get_coords(const HexBasis *pHex, vector<double> &coords, const T& value,
		  const CellData &cd) const;
 
protected:
  //! find a reasonable initial guess for starting Newton iteration.
  template <class CellData>
  void initial_guess(const HexBasis *pHex, const T &val, const CellData &cd, 
		     vector<double> cur) const
  {
    double dist = HUGE;
	
    int end = 3;
    vector<double> guess(3,0.L);

    for (int z = 0; z <= end; z++) {
      guess[2] = z / (double) end;
      for (int y = 0; y <= end; y++) {
	guess[1] = y / (double) end;
	for (int x = 0; x <= end; x++) {
	  guess[0] = x / (double) end;
	  T i_val = pHex->interpolate(guess, cd);
	  T dv = val_type_difference(i_val, val);
	  double cur_d = val_type_length(dv);
	  if (cur_d < dist) {
	    dist = cur_d;
	    cur = guess;
	  }
	}
      }
    }
  }
};

template <class T>
void
eval_guess(vector<double> &cur, vector<double> &last, double &dist, 
	   const T &i_val, const T& value, const vector<T> &grad); 

template <>
void 
eval_guess<Point>(vector<double> &cur, vector<double> &last, double &dist,
		  const Point &i_val, const Point& value, 
		  const vector<Point> &grad);

// meant to compile for scalars
template <class T>
void 
eval_guess(vector<double> &cur, vector<double> &last, double &dist, 
	 const T& i_val, const T& value, const vector<T>& grad)
{
  T dif = i_val - value; 
  cur[0] += dif / grad[0];
  dist = (last - cur).length();
}


template <class HexBasis>
template <class CellData>
void 
HexLocate<HexBasis>::get_coords(const HexBasis *pHex, vector<double> &coords, 
				const T& value, const CellData &cd) const  
{     
  //! Step 1: get a good guess on the domain, evaluate equally spaced points 
  //!         on the domain and use the closest as our starting point for 
  //!         Newton iteration.  dist=(last-cur).length();

      
  vector<double> cur(3), last(3); 
  initial_guess(pHex, value, cd, cur);
      
  //! Now cur has our initialization param for Newton iteration.
  //! Step 2: Newton iteration.
  //! f(u) =C'(u) dot (C(u) - P)
  //! the distance from P to C(u) is minimum when f(u) = 0 
  double dist;
  do {
    last = cur;
    T i_val = pHex->interpolate(cur, cd);
    vector<T> grad(3);
    pHex->derivate(cur, cd, grad);
    eval_guess(cur, last, dist, i_val, value, grad);
  } while (dist > 0.00001);
  coords.clear();
  coords = cur;
};
 


template <class T>
class HexTrilinearLgn : public HexApprox
{
public:
  typedef T value_type;

  HexTrilinearLgn() {}
  virtual ~HexTrilinearLgn() {}
  
  int polynomial_order() const { return 1; }

  static int GaussianNum;
  static double GaussianPoints[8][3];
  static double GaussianWeights[8];
 
  // Value at coord
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
  
  //! First derivative at coord.
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<double> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];

    derivs.size(3);

    derivs[0]=(-1 + y + z - y*z)*cd.node0()
      +(-1 +y)*(-1 + z)*cd.node1()
      +(y - y*z)*cd.node2()
      +y*(-1 + z)*cd.node3()
      +(-1 + y)*z*cd.node4()
      +(z - y*z)*cd.node5()
      +y*z*cd.node6()
      -(y*z)*cd.node7();
      
    derivs[1]=(-1 + x + z - x*z)*cd.node0()
      +x*(-1 + z)*cd.node1()
      +(x - x*z)*cd.node2()
      +(-1 + x)*(-1 + z)*cd.node3()
      +(-1 + x)*z*cd.node4()
      -(x*z)*cd.node5()
      +x*z*cd.node6()
      +(z - x*z)*cd.node7();
      
    derivs[2]=(-1 + x + y - x*y)*cd.node0()
      +x*(-1 + y)*cd.node1()
      -(x*y)*cd.node2()
      +(-1 + x)*y*cd.node3()
      +(-1 + x)*(-1 + y)*cd.node4()
      +(x - x*y)*cd.node5()
      +x*y*cd.node6()
      +(y - x*y)*cd.node7();
  }
  
  //! return the parametric coordinates for value within the element.
  template <class CellData>
  void get_coords(vector<double> &coords, const T& value, 
		  const CellData &cd) const  
  {
    HexLocate< HexTrilinearLgn<T> > CL;
    CL.get_coords(this, coords, value, cd);
  };
    
  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);

protected:
  //! Find a reasonably close starting set of parametric coordinates, 
  //! to val.
  template <class ElemData>
  void initial_guess(vector<double> &coords, const T &val, 
		     const ElemData &cd) const;

  //! next_guess is the next Newton iteration step.
  template <class ElemData>
  void next_guess(vector<double> &coords, const T &val, 
		  const ElemData &cd) const;

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
  
template <class T>
int HexTrilinearLgn<T>::GaussianNum = 8;
  
template <class T>
double HexTrilinearLgn<T>::GaussianPoints[8][3] = {
  {0.211324865405, 0.211324865405, 0.211324865405},
  {0.788675134595, 0.211324865405, 0.211324865405},
  {0.788675134595, 0.788675134595, 0.211324865405},
  {0.211324865405, 0.788675134595, 0.211324865405},
  {0.211324865405, 0.211324865405, 0.788675134595},
  {0.788675134595, 0.211324865405, 0.788675134595},
  {0.788675134595, 0.788675134595, 0.788675134595},
  {0.211324865405, 0.788675134595, 0.788675134595}};
  
template <class T>
double HexTrilinearLgn<T>::GaussianWeights[8] = 
  {.125, .125, .125, .125, .125, .125, .125, .125};
  
} //namespace SCIRun

#endif // HexTrilinear_h
