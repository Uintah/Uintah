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
//    File   : Locate.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Oct 08 2005

#if !defined(Locate_h)
#define Locate_h

#include <vector>
#include <math.h>

#include <iostream>
#include <fstream>

#include <Core/Geometry/Point.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
// Turn off 'implicit conversion... loss of accuracy' messages.
#  pragma set woff 1506
#endif

#include <Core/Basis/share.h>

namespace SCIRun {

  using std::vector;
  using std::cerr;
  using std::endl;

  template<class T>
    inline T InverseMatrix3x3(const T *p, T *q) 
  {
    const T a=p[0], b=p[1], c=p[2];
    const T d=p[3], e=p[4], f=p[5];
    const T g=p[6], h=p[7], i=p[8];
      
    const T detp=a*e*i-c*e*g+b*f*g+c*d*h-a*f*h-b*d*i;
    const T detinvp=(detp ? 1.0/detp : 0);
      
    q[0]=(e*i-f*h)*detinvp;
    q[1]=(c*h-b*i)*detinvp;
    q[2]=(b*f-c*e)*detinvp;
    q[3]=(f*g-d*i)*detinvp;
    q[4]=(a*i-c*g)*detinvp;
    q[5]=(c*d-a*f)*detinvp;
    q[6]=(d*h-e*g)*detinvp;
    q[7]=(b*g-a*h)*detinvp;
    q[8]=(a*e-b*d)*detinvp;
  
    return detp;
  }

  template<class T>
    inline T DetMatrix3x3(const T *p) 
  {
    const T a=p[0], b=p[1], c=p[2];
    const T d=p[3], e=p[4], f=p[5];
    const T g=p[6], h=p[7], i=p[8];
      
    const T detp=a*e*i-c*e*g+b*f*g+c*d*h-a*f*h-b*d*i;
    return detp;
  }

  //! default case for volume calculation - currently not needed
  template <class T>
    inline double d_volume(const vector<T>& derivs)
  {
    ASSERT(0); // to do
    return 0;  
  }

  template <>
    inline double d_volume(const vector<Point>& derivs)
  {
    double J[9];

    J[0]=derivs[0].x();
    J[3]=derivs[0].y();
    J[6]=derivs[0].z();
    J[1]=derivs[1].x(); 
    J[4]=derivs[1].y();
    J[7]=derivs[1].z();
    J[2]=derivs[2].x();
    J[5]=derivs[2].y();
    J[8]=derivs[2].z();

    return DetMatrix3x3(J);
  }

  template <class ElemBasis, class ElemData>
    double get_volume3(const ElemBasis *pEB, const ElemData &cd)
  {
    double volume=0.;
  
    vector<double> coords(3);
    for(int i=0; i<ElemBasis::GaussianNum; i++) {
      coords[0]=ElemBasis::GaussianPoints[i][0];
      coords[1]=ElemBasis::GaussianPoints[i][1];
      coords[2]=ElemBasis::GaussianPoints[i][2];
 
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
      volume+=ElemBasis::GaussianWeights[i]*d_volume(derivs);
    }
    return volume*pEB->volume();
  }
 


  //! default case for face area calculation - currently not needed
  template <class T>
    inline double d_area(const vector<T>& derivs, const vector<double>& dv0, const vector<double>& dv1)
  {
    ASSERT(0); // to do
    return 0;  
  }

  //! area calculation on points
  template <>
    inline double d_area(const vector<Point>& derivs, const vector<double>& dv0, const vector<double>& dv1)
  {
    const unsigned int dvsize=derivs.size();
    ASSERT(dv0.size()==dvsize);
    ASSERT(dv1.size()==dvsize);

    Vector Jdv0(0,0,0), Jdv1(0,0,0);
    for(unsigned int i = 0; i<dvsize; i++) {
      Jdv0+=dv0[i]*Vector(derivs[i].x(), derivs[i].y(), derivs[i].z());
      Jdv1+=dv1[i]*Vector(derivs[i].x(), derivs[i].y(), derivs[i].z());
    }
  
    return Cross(Jdv0, Jdv1).length();
  }


  template <class NumApprox, class ElemBasis, class ElemData>
    double get_area2(const ElemBasis *pEB, const unsigned face, 
		     const ElemData &cd)
  {
    const double *v0 = pEB->unit_vertices[pEB->unit_faces[face][0]];
    const double *v1 = pEB->unit_vertices[pEB->unit_faces[face][1]];
    const double *v2 = pEB->unit_vertices[pEB->unit_faces[face][2]];
    vector<double> d0(2), d1(2);
    d0[0]=v1[0]-v0[0];
    d0[1]=v1[1]-v0[1];
    d1[0]=v2[0]-v0[0];
    d1[1]=v2[1]-v0[1];
    double area=0.;
  
    vector<double> coords(2);
    for(int i=0; i<NumApprox::GaussianNum; i++) {
      coords[0]=v0[0]+NumApprox::GaussianPoints[i][0]*d0[0]+NumApprox::GaussianPoints[i][1]*d1[0];
      coords[1]=v0[1]+NumApprox::GaussianPoints[i][0]*d0[1]+NumApprox::GaussianPoints[i][1]*d1[1];
 
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
 /*      cerr << "D "; */
/*       for(int j=0; j<derivs.size(); j++) */
/* 	cerr << derivs[j] << " "; */
/*       cerr << endl; */
/*       cerr << "w, al " << NumApprox::GaussianWeights[i] << " " << d_area(derivs, d0, d1) << endl; */
      area+=NumApprox::GaussianWeights[i]*d_area(derivs, d0, d1);
    }
    return area*pEB->area(face);
  }
 
  template <class NumApprox, class ElemBasis, class ElemData>
    double get_area3(const ElemBasis *pEB, const unsigned face, 
		     const ElemData &cd)
  {
    const double *v0 = pEB->unit_vertices[pEB->unit_faces[face][0]];
    const double *v1 = pEB->unit_vertices[pEB->unit_faces[face][1]];
    const double *v2 = pEB->unit_vertices[pEB->unit_faces[face][2]];
    vector<double> d0(3), d1(3);
    d0[0]=v1[0]-v0[0];
    d0[1]=v1[1]-v0[1];
    d0[2]=v1[2]-v0[2];
    d1[0]=v2[0]-v0[0];
    d1[1]=v2[1]-v0[1];
    d1[2]=v2[2]-v0[2];
    double area=0.;
  
    vector<double> coords(3);
    for(int i=0; i<NumApprox::GaussianNum; i++) {
      coords[0]=v0[0]+NumApprox::GaussianPoints[i][0]*d0[0]+NumApprox::GaussianPoints[i][1]*d1[0];
      coords[1]=v0[1]+NumApprox::GaussianPoints[i][0]*d0[1]+NumApprox::GaussianPoints[i][1]*d1[1];
      coords[2]=v0[2]+NumApprox::GaussianPoints[i][0]*d0[2]+NumApprox::GaussianPoints[i][1]*d1[2];
 
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
/*       cerr << "D "; */
/*       for(int j=0; j<derivs.size(); j++) */
/* 	cerr << derivs[j] << " "; */
/*       cerr << endl; */
/*       cerr << "w, al " << NumApprox::GaussianWeights[i] << " " << d_area(derivs, d0, d1) << endl; */
      area+=NumApprox::GaussianWeights[i]*d_area(derivs, d0, d1);
    }
    return area*pEB->area(face);
  }
 
  //! default case for arc length calculation - currently not needed
  template <class T>
    inline double d_arc_length(const vector<T>& derivs, const vector<double>& dv)
  {
    ASSERT(0); // to do
    return 0;  
  }

  //! arc length calculation on points
  template <>
    inline double d_arc_length(const vector<Point>& derivs, const vector<double>& dv)
  {
    const unsigned int dvsize=dv.size();
    ASSERT(derivs.size()==dvsize);
    double Jdv[3];
    Jdv[0]=Jdv[1]=Jdv[2]=0.;
    for(unsigned int i = 0; i<dvsize; i++) {
      Jdv[0]+=dv[i]*derivs[i].x();
      Jdv[1]+=dv[i]*derivs[i].y();
      Jdv[2]+=dv[i]*derivs[i].z();
    }
  
    return sqrt(Jdv[0]*Jdv[0]+Jdv[1]*Jdv[1]+Jdv[2]*Jdv[2]);
  }

  template <class NumApprox, class ElemBasis, class ElemData>
    double get_arc1d_length(const ElemBasis *pEB, const unsigned edge, 
			    const ElemData &cd)
  {
    const double *v0 = pEB->unit_vertices[pEB->unit_edges[edge][0]];
    const double *v1 = pEB->unit_vertices[pEB->unit_edges[edge][1]];
    vector<double> dv(1);
    dv[0]=v1[0]-v0[0];
    double arc_length=0.;
  
    vector<double> coords(1);
    for(int i=0; i<NumApprox::GaussianNum; i++) {
      coords[0]=v0[0]+NumApprox::GaussianPoints[i][0]*dv[0];
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
/*       cerr << "D "; */
/*       for(int j=0; j<derivs.size(); j++) */
/* 	cerr << derivs[j] << " "; */
/*       cerr << endl; */
/*       cerr << "w, al " << NumApprox::GaussianWeights[i] << " " << d_arc_length(derivs, dv) << endl; */
      arc_length+=NumApprox::GaussianWeights[i]*d_arc_length(derivs, dv);
    }
    return arc_length;
  }
 

  template <class NumApprox, class ElemBasis, class ElemData>
    double get_arc2d_length(const ElemBasis *pEB, const unsigned edge, 
			    const ElemData &cd)
  {
    const double *v0 = pEB->unit_vertices[pEB->unit_edges[edge][0]];
    const double *v1 = pEB->unit_vertices[pEB->unit_edges[edge][1]];
    vector<double> dv(2);
    dv[0]=v1[0]-v0[0];
    dv[1]=v1[1]-v0[1];
    double arc_length=0.;
  
    vector<double> coords(2);
    for(int i=0; i<NumApprox::GaussianNum; i++) {
      coords[0]=v0[0]+NumApprox::GaussianPoints[i][0]*dv[0];
      coords[1]=v0[1]+NumApprox::GaussianPoints[i][0]*dv[1];
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
      //   cerr << "D ";
      //     for(int j=0; j<derivs.size(); j++)
      //      cerr << derivs[j] << " ";
      // cerr << endl;
      //    cerr << "w, al " << NumApprox::GaussianWeights[i] << " " << d_arc_length(derivs, dv) << endl;
      arc_length+=NumApprox::GaussianWeights[i]*d_arc_length(derivs, dv);
    }
    return arc_length;
  }
 

  template <class NumApprox, class ElemBasis, class ElemData>
    double get_arc3d_length(const ElemBasis *pEB, const unsigned edge, 
			    const ElemData &cd)
  {
    const double *v0 = pEB->unit_vertices[pEB->unit_edges[edge][0]];
    const double *v1 = pEB->unit_vertices[pEB->unit_edges[edge][1]];
    vector<double> dv(3);
    dv[0]=v1[0]-v0[0];
    dv[1]=v1[1]-v0[1];
    dv[2]=v1[2]-v0[2];
    double arc_length=0.;
  
    vector<double> coords(3);
    for(int i=0; i<NumApprox::GaussianNum; i++) {
      coords[0]=v0[0]+NumApprox::GaussianPoints[i][0]*dv[0];
      coords[1]=v0[1]+NumApprox::GaussianPoints[i][0]*dv[1];
      coords[2]=v0[2]+NumApprox::GaussianPoints[i][0]*dv[2];
      vector<typename ElemBasis::value_type> derivs;
      pEB->derivate(coords, cd, derivs);
      //    cerr << "w, al " << NumApprox::GaussianWeights[i] << " " << d_arc_length(derivs, dv) << endl;
      arc_length+=NumApprox::GaussianWeights[i]*d_arc_length(derivs, dv);
    }
    return arc_length;
  }
 

  //default case
  template <class T>
    T difference(const T& interp, const T& value)
  {
    return interp - value;
  }

  template <>
    SHARE Point difference(const Point& interp, const Point& value);
 


  template <class T>
    double getnextx1(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd);

  template <>
    SHARE double getnextx1(vector<double> &x, vector<double> &xold, 
		     const Point& y, const vector<Point>& yd);

  template <class T>
    double getnextx1(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd)
  {
    x[0] -= (yd[0] ? y/yd[0] : 0.);
    const double dx=x[0]-xold[0];
    return sqrt(dx*dx);
  }

  template <class T>
    double getnextx2(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd);

  template <>
    SHARE double getnextx2(vector<double> &x, vector<double> &xold, 
		     const Point& y, const vector<Point>& yd);

  template <class T>
    double getnextx2(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd)
  {
    x[0] -= (yd[0] ? y/yd[0] : 0.);
    x[1] -= (yd[1] ? y/yd[1] : 0.);
    const double dx=x[0]-xold[0];
    const double dy=x[1]-xold[1];
    return sqrt(dx*dx+dy*dy);

  }

  // locate for scalar value 
  template <class T>
    double getnextx3(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd);
 
  // locate for Point 
  template <>
    SHARE double getnextx3(vector<double> &x, vector<double> &xold, 
		     const Point& y, const vector<Point>& yd);
      
  // locate for scalar value 
  template <class T>
    double getnextx3(vector<double> &x, vector<double> &xold, 
		     const T& y, const vector<T>& yd)
  {
    x[0] -= (yd[0] ? y/yd[0] : 0.);
    x[1] -= (yd[1] ? y/yd[1] : 0.);
    x[2] -= (yd[2] ? y/yd[2] : 0.);

    const double dx=x[0]-xold[0];
    const double dy=x[1]-xold[1];
    const double dz=x[2]-xold[2];
    return sqrt(dx*dx+dy*dy+dz*dz);	
  }



  //! Class for searching of parametric coordinates related to a 
  //! value in 3d meshes and fields
  //! More general function: find value in interpolation for given value
  //! Step 1: get a good guess on the domain, evaluate equally spaced points 
  //!         on the domain and use the closest as our starting point for 
  //!         Newton iteration. (implemented in derived class)
  //! Step 2: Newton iteration.
  //!         x_n+1 =x_n + y(x_n) * y'(x_n)^-1       

  template <class ElemBasis>
class Dim3Locate {
  public:
    typedef typename ElemBasis::value_type T;
    static const double thresholdDist; //!< Thresholds for coordinates checks
    static const double thresholdDist1; //!< 1+thresholdDist
    static const int maxsteps; //!< maximal steps for Newton search

    Dim3Locate() {}
    virtual ~Dim3Locate() {}
 
    //! find value in interpolation for given value
    template <class ElemData>
      bool get_iterative(const ElemBasis *pEB, vector<double> &x, 
			 const T& value, const ElemData &cd) const  
    {       
      vector<double> xold(3); 
      vector<T> yd(3);
      for (int steps=0; steps<maxsteps; steps++) {
	xold = x;
	T y = difference(pEB->interpolate(x, cd), value);
	pEB->derivate(x, cd, yd);
	double dist=getnextx3(x, xold, y, yd);
	if (dist < thresholdDist) 
	  return true;	  
      }
      return false;
    }
  };

  template<class ElemBasis>
    const double Dim3Locate<ElemBasis>::thresholdDist=1e-7;
  template<class ElemBasis>
    const double Dim3Locate<ElemBasis>::thresholdDist1=1.+Dim3Locate::thresholdDist;
  template<class ElemBasis>
    const int Dim3Locate<ElemBasis>::maxsteps=100;
  
  //! Class for searching of parametric coordinates related to a 
  //! value in 2d meshes and fields
  template <class ElemBasis>
class Dim2Locate {
  public:
    typedef typename ElemBasis::value_type T;
    static const double thresholdDist; //!< Thresholds for coordinates checks
    static const double thresholdDist1; //!< 1+thresholdDist
    static const int maxsteps; //!< maximal steps for Newton search

    Dim2Locate() {}
    virtual ~Dim2Locate() {}
 
    //! find value in interpolation for given value
    template <class ElemData>
      bool get_iterative(const ElemBasis *pEB, vector<double> &x, 
			 const T& value, const ElemData &cd) const  
    {       
      vector<double> xold(2); 
      vector<T> yd(2);
      for (int steps=0; steps<maxsteps; steps++) {
	xold = x;
	T y = difference(pEB->interpolate(x, cd), value);
	pEB->derivate(x, cd, yd);
	double dist=getnextx2(x, xold, y, yd);
	if (dist < thresholdDist) 
	  return true;	  
      }
      return false;
    }
  };

  template<class ElemBasis>
    const double Dim2Locate<ElemBasis>::thresholdDist=1e-7;
  template<class ElemBasis>
    const double Dim2Locate<ElemBasis>::thresholdDist1=
    1.+Dim2Locate::thresholdDist;
  template<class ElemBasis>
    const int Dim2Locate<ElemBasis>::maxsteps=100;
  

  //! Class for searching of parametric coordinates related to a 
  //! value in 1d meshes and fields
  template <class ElemBasis>
class Dim1Locate {
  public:

    typedef typename ElemBasis::value_type T;
    static const double thresholdDist; //!< Thresholds for coordinates checks
    static const double thresholdDist1; //!< 1+thresholdDist
    static const int maxsteps; //!< maximal steps for Newton search

    Dim1Locate() {}
    virtual ~Dim1Locate() {}

    //! find value in interpolation for given value         
    template <class ElemData>
      bool get_iterative(const ElemBasis *pElem, vector<double> &x, 
			 const T& value, const ElemData &cd) const  
    {          
      vector<double> xold(1); 
      vector<T> yd(1);
 	
      for (int steps=0; steps<maxsteps; steps++) {
	xold = x;
	T y = difference(pElem->interpolate(x, cd), value); 
	pElem->derivate(x, cd, yd);
	double dist=getnextx1(x, xold, y, yd);
	if (dist < thresholdDist)
	  return true;
      }
      return false;
    }
  };

  template<class ElemBasis>
    const double Dim1Locate<ElemBasis>::thresholdDist=1e-7;
  template<class ElemBasis>
    const double Dim1Locate<ElemBasis>::thresholdDist1=1.+Dim1Locate::thresholdDist;
  template<class ElemBasis>
    const int Dim1Locate<ElemBasis>::maxsteps=100;




  // default case compiles for scalar types
  template <class T>
    bool compare_distance(const T &interp, const T &val, 
			  double &cur_d, double dist) 
  {
    double dv = interp - val;
    cur_d = sqrt(dv*dv);
    return  cur_d < dist;
  }

  template <>
    SHARE bool compare_distance(const Point &interp, const Point &val, 
			  double &cur_d, double dist);

  template <class T>
    bool check_zero(const vector<T> &val) 
  {
    typename vector<T>::const_iterator iter = val.begin();
    while(iter != val.end()) {
      const T &v=*iter++;
      if (fabs(v)>1e-7)
	return false;
    }
    return true;
  }

  template <>
    SHARE bool check_zero(const vector<Point> &val); 

}
#endif
