/*
 *  CubicPWI.h: cubic piecewise interpolation
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef SCI_CUBICPWI_H__
#define SCI_CUBICPWI_H__

#include <SCICore/Math/PiecewiseInterp.h>


#include <SCICore/share/share.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

#include <iostream>

namespace SCICore {
namespace Math {

enum SplineMode {CATMULL};

using SCICore::Containers::Array1;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

using std::cout;
using std::endl;

template <class T> SCICORESHARE std::ostream& operator<<(std::ostream& out, Array1<T> a){
  for (int i=0; i<a.size(); i++){
    std::cout << a[i] << std::endl;
  }
  return out;
}

typedef struct Quat {
  double a;
  double b;
  double c;
  double d;
} QUAT;

class SCICORESHARE CubicPWI: public PiecewiseInterp<double> {
public:
  CubicPWI();
  CubicPWI(const Array1<double>&, const Array1<double>&);
  
  bool set_data(const Array1<double>&, const Array1<double>&);
  inline bool get_value(double, double&);

private:
  Array1<QUAT> p;
};

inline bool CubicPWI::get_value(double w, double& res){
  int i;
  if (data_valid && (i=get_interval(w))>=0){
    res=p[i].a+w*(p[i].b+w*(p[i].c+p[i].d*w));
    return true;
  }
  else
   return false;
}

template <class T> class SCICORESHARE Cubic3DPWI: public PiecewiseInterp<T> {
public:
  Cubic3DPWI() {};
  Cubic3DPWI(const Array1<double>&, const Array1<T>&);
  
  bool set_data(const Array1<double>&, const Array1<T>&);
  bool set_data(const Array1<double>&, const Array1<T>&, const Array1<Vector>&);
  inline bool get_value(double, T&);
  
private:
  Array1<QUAT> X;
  Array1<QUAT> Y;
  Array1<QUAT> Z;
};

template <class T> Cubic3DPWI<T>::Cubic3DPWI(const Array1<double>& pts, const Array1<T>& vals) {
  set_data(pts, vals);
}

template <class T> inline bool Cubic3DPWI<T>::get_value(double w, T& res){
  int i;
  if (data_valid && (i=get_interval(w))>=0){
    res=T(X[i].a+w*(X[i].b+w*(X[i].c+X[i].d*w)), 
	  Y[i].a+w*(Y[i].b+w*(Y[i].c+Y[i].d*w)),
	  Z[i].a+w*(Z[i].b+w*(Z[i].c+Z[i].d*w)));
    return true;
  }
  else
    return false;
}

bool SCICORESHARE first_drv(const Array1<double>&, const Array1<double>&, Array1<double>&, int tns=0);

template <class T> bool Cubic3DPWI<T>::set_data(const Array1<double>& pts, const Array1<T>& vals){
  int sz=vals.size();
  reset();
  Array1<double> drvX, drvY, drvZ;
  Array1<double> vx, vy, vz;
  vx.resize(sz);
  vy.resize(sz);
  vz.resize(sz);
  for (int i=0; i<sz; i++){
      vx[i]=vals[i].x();
      vy[i]=vals[i].y();
      vz[i]=vals[i].z();
  }

  if (first_drv(pts, vx, drvX, 0) && first_drv(pts, vy, drvY, 0) 
      && first_drv(pts, vz, drvZ, 0)) {
    cout << "Derivatives are done!!!" << endl;
   Array1<Vector> drvs;
    drvs.resize(sz);
    for (int i=0; i<sz; i++)
      drvs[i]=Vector(vx[i], vy[i], vz[i]);
    return set_data(pts, vals, drvs); 		   
  }
  else {
    return false;
  }
}

// takes sorted array of points
template <class T> bool Cubic3DPWI<T>::set_data(const Array1<double>& pts, const Array1<T>& vals, const Array1<Vector>& drvs){
  int sz=0;
  reset();
  
//   cout << "About to fill in data in set_data!!!" << endl;
//   cout << "pts-array:" << endl;
//   cout << pts;

 //  cout << "vals-array:" << endl;
//   cout << vals;
  
//   cout << "drvs-array:" << endl;
//   cout << drvs;
//   cout << "sizes" << endl;
//   cout <<points.size()  << endl << vals.size() << endl << drvs.size()<< endl;
  if (fill_data(pts) && (sz=points.size())>1 && sz==vals.size() && sz==drvs.size()){
    cout << "Inside set_data!!!" << endl;
    X.resize(sz);
    Y.resize(sz);
    Z.resize(sz);
    
    Array1<double> drvX, drvY, drvZ;
    Array1<double> vx, vy, vz;
    vx.resize(sz);
    vy.resize(sz);
    vz.resize(sz);
    drvX.resize(sz);
    drvY.resize(sz);
    drvZ.resize(sz);
    
    for (int i=0; i<sz; i++){
      vx[i]=vals[i].x();
      vy[i]=vals[i].y();
      vz[i]=vals[i].z();
      drvX[i]=drvs[i].x();
      drvY[i]=drvs[i].y();
      drvZ[i]=drvs[i].z();
    }

    /* cout << "vx-array:" << endl;
    cout << vx;
    
    cout << "vfy-array:" << endl;
    cout << vy;

    cout << "vz-array:" << endl;
    cout << vz;
    */
   
//     cout << "First Derivatives are:" << endl;
//     cout << "x-dir:" << drvX << endl;
//     cout << "y-dir:" << drvY << endl;
//     cout << "z-dir:" << drvZ << endl;
    
    data_valid=true;
    double tmp, delta, a, b, c, d;
    
    for (int i=0; i<sz-1; i++)
      if ( (delta=points[i+1]-points[i]) >10e-9){
	
	tmp=points[i];
	
	a=vx[i];
	b=drvX[i];
	c=((3*(vx[i+1]-vx[i])/delta)-2*drvX[i]-drvX[i+1])/delta;
	d=(drvX[i]+drvX[i+1]-(2*(vx[i+1]-vx[i])/delta))/(delta*delta);
	
	X[i].a=a+tmp*(tmp*(c-tmp*d)-b);
	X[i].b=b-tmp*(2*c-3*tmp*d);
	X[i].c=c-3*tmp*d;
	X[i].d=d;
	
	cout << "Interval: " << points[i] << ", " << points[i+1] << endl;
	cout << "Coeff. are for X: " << X[i].a << endl << X[i].b << endl << X[i].c << endl << X[i].d << endl;
	
	a=vy[i];
	b=drvY[i];
	c=(3*(vy[i+1]-vy[i])/delta-2*drvY[i]-drvY[i+1])/delta;
	d=(drvY[i]+drvY[i+1]-2*(vy[i+1]-vy[i])/delta)/(delta*delta);
	
	Y[i].a=a+tmp*(tmp*(c-tmp*d)-b);
	Y[i].b=b-tmp*(2*c-3*tmp*d);
	Y[i].c=c-3*tmp*d;
	Y[i].d=d;
	
	cout << "Interval: " << points[i] << ", " << points[i+1] << endl;
	cout << "Coeff. are for Y: " << Y[i].a << endl << Y[i].b << endl << Y[i].c << endl << Y[i].d << endl;
	
	a=vz[i];
	b=drvZ[i];
	c=(3*(vz[i+1]-vz[i])/delta-2*drvZ[i]-drvZ[i+1])/delta;
	d=(drvZ[i]+drvZ[i+1]-2*(vz[i+1]-vz[i])/delta)/(delta*delta);
	
	Z[i].a=a+tmp*(tmp*(c-tmp*d)-b);
	Z[i].b=b-tmp*(2*c-3*tmp*d);
	Z[i].c=c-3*tmp*d;
	Z[i].d=d;
	
// 	cout << "Interval: " << points[i] << ", " << points[i+1] << endl;
// 	cout << "Coeff. are for Z: " << Z[i].a << endl << Z[i].b << endl << Z[i].c << endl << Z[i].d << endl;
      }
      else {
	cout << "Delta is small!!! " << endl;
	reset();
	break;
      }
  }
  return data_valid;
}

} // Math 
} // SCICore

#endif //SCI_CUBICPWI_H__















