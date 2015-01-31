/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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



#ifndef SCI_CUBICPWI_H__
#define SCI_CUBICPWI_H__

#include <Core/Math/PiecewiseInterp.h>


#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include   <iostream>

namespace SCIRun {

enum EndCondition {natural_ends, clamped_ends, bessel_ends, quadratic_ends};

template <class T> class Cubic3DPWI;

class CubicPWI: public PiecewiseInterp<double>
{
public:
  CubicPWI();
  CubicPWI(const Array1<double>&, const Array1<double>&);
  
  bool set_data(const Array1<double>&, const Array1<double>&);
  inline bool get_value(double, double&);

  // Quat struct should be private to this class, but the gcc 3.2.3
  // compiler then won't let friend classes use it... sigh... so
  // making it public for now.
  struct Quat {
    double a;
    double b;
    double c;
    double d;
  };

private:
  
  Array1<Quat> p;
};

inline bool CubicPWI::get_value(double w, double& res)
{
  int i = get_interval(w);
  if( data_valid && i >= 0 ) {
    w -= points[i];
    res = p[i].a + w * (p[i].b+w*(p[i].c+p[i].d*w));
    return true;
  }
  else
    return false;
}

template <class T> class Cubic3DPWI: public PiecewiseInterp<T> {
public:
  Cubic3DPWI() {};
  Cubic3DPWI(const Array1<double>&, const Array1<T>&);
  
  bool set_data(const Array1<double>&, const Array1<T>&);
  bool set_data(const Array1<double>&, const Array1<T>&, 
		const Array1<Vector>&);
  inline bool get_value(double, T&);
  
private:
  
  CubicPWI p;

  Array1<CubicPWI::Quat> X;
  Array1<CubicPWI::Quat> Y;
  Array1<CubicPWI::Quat> Z;
};

template <class T> Cubic3DPWI<T>::Cubic3DPWI(const Array1<double>& pts, 
					     const Array1<T>& vals) {
  set_data(pts, vals);
}

template <class T> inline bool Cubic3DPWI<T>::get_value(double w, T& res){
  int i;
  if (this->data_valid && (i = this->get_interval(w))>=0){
    w -= this->points[i];
    res=T(X[i].a+w*(X[i].b+w*(X[i].c+X[i].d*w)), 
	  Y[i].a+w*(Y[i].b+w*(Y[i].c+Y[i].d*w)),
	  Z[i].a+w*(Z[i].b+w*(Z[i].c+Z[i].d*w)));
    return true;
  }
  else
    return false;
}


bool set_tangents(const Array1<double>&, const Array1<double>&,
			       Array1<double>&, EndCondition);

template <class T> bool 
Cubic3DPWI<T>::set_data(const Array1<double>& pts, const Array1<T>& vals){
  int sz=vals.size();
  this->reset();
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

  if (set_tangents(pts, vx, drvX, natural_ends) && 
      set_tangents(pts, vy, drvY, natural_ends) &&
      set_tangents(pts, vz, drvZ, natural_ends)) {
    // cout << "Derivatives are done!!!" << "\n";
    Array1<Vector> drvs;
    drvs.resize(sz);
    for (int i=0; i<sz; i++)
      drvs[i]=Vector(drvX[i], drvY[i], drvZ[i]);
    return set_data(pts, vals, drvs); 		   
  }
  else {
    return false;
  }
}

// takes sorted array of points
template <class T> bool 
Cubic3DPWI<T>::set_data(const Array1<double>& pts, const Array1<T>& vals, 
			const Array1<Vector>& drvs){
  int sz=0;
  this->reset();

  if( this->fill_data(pts) && (sz = this->points.size()) > 1 && 
      sz == vals.size() && sz == drvs.size() ) {
    // cout << "Inside set_data!!!" << "\n";
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
    
    this->data_valid=true;
    double delta, a, b, c, d;
    
    for (int i=0; i<sz-1; i++)
      if ( (delta = this->points[i+1] - this->points[i]) >10e-9){
	
	a=vx[i];
	b=drvX[i];
	c=((3*(vx[i+1]-vx[i])/delta)-2*drvX[i]-drvX[i+1])/delta;
	d=(drvX[i]+drvX[i+1]-(2*(vx[i+1]-vx[i])/delta))/(delta*delta);
	
	X[i].a=a;
	X[i].b=b;
	X[i].c=c;
	X[i].d=d;
	
	// cout << "Interval: " << this->points[i] << ", " << this->points[i+1] << "\n";
	// cout << "Coeff. are for X: " << X[i].a << "\n" << X[i].b 
	//      << "\n" << X[i].c << "\n" << X[i].d << "\n";
	
	a=vy[i];
	b=drvY[i];
	c=(3*(vy[i+1]-vy[i])/delta-2*drvY[i]-drvY[i+1])/delta;
	d=(drvY[i]+drvY[i+1]-2*(vy[i+1]-vy[i])/delta)/(delta*delta);
	
	Y[i].a=a;
	Y[i].b=b;
	Y[i].c=c;
	Y[i].d=d;

	// cout << "Interval: " << this->points[i] << ", " << this->points[i+1] << "\n";
	// cout << "Coeff. are for Y: " << Y[i].a << "\n" << Y[i].b 
	//      << "\n" << Y[i].c << "\n" << Y[i].d << "\n";
	
	a=vz[i];
	b=drvZ[i];
	c=(3*(vz[i+1]-vz[i])/delta-2*drvZ[i]-drvZ[i+1])/delta;
	d=(drvZ[i]+drvZ[i+1]-2*(vz[i+1]-vz[i])/delta)/(delta*delta);
	
	Z[i].a=a;
	Z[i].b=b;
	Z[i].c=c;
	Z[i].d=d;
      }
      else {
	std::cout << "Delta is small!!!\n";
	this->reset();
	break;
      }
  }
  return this->data_valid;
}

template <class T> std::ostream& 
operator<<(std::ostream& out, Array1<T> a){
  for (int i=0; i<a.size(); i++){
    std::cout << a[i] << "\n";
  }
  return out;
}


} // End namespace SCIRun

#endif //SCI_CUBICPWI_H__


