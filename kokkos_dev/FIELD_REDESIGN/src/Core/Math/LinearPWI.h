/*
 *  LinearPWI.h: linear piecewise interpolation templates
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef SCI_LINEARPWI_H__
#define SCI_LINEARPWI_H__

#include <SCICore/Math/PiecewiseInterp.h>

#include <SCICore/share/share.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
namespace Math {

using namespace SCICore::Containers;
using SCICore::Geometry::Point;

typedef struct Pair{
  double a;
  double b; 
} PAIR;


class SCICORESHARE LinearPWI: public PiecewiseInterp<double> {
public:
	LinearPWI();
	LinearPWI(const Array1<double>&, const Array1<double>&);

	bool set_data(const Array1<double>&, const Array1<double>&);
	inline bool get_value(double, double&);

private:
	Array1<PAIR> p;
};

inline bool LinearPWI::get_value(double w, double& res){
  int interv;
  if (data_valid && (interv=get_interval(w))>=0){
    res=p[interv].a+p[interv].b*w;
    return true;
  }
  else
    return false;
}

template <class T> class SCICORESHARE Linear3DPWI: public PiecewiseInterp<T> {
public:
  Linear3DPWI() {};
  Linear3DPWI(const Array1<double>&, const Array1<T>&);
  
  bool set_data(const Array1<double>&, const Array1<T>&);
  inline bool get_value(double, T&);
  
private:
  Array1<PAIR> X;
  Array1<PAIR> Y;
  Array1<PAIR> Z;
};


template <class T> Linear3DPWI<T>::Linear3DPWI(const Array1<double>& pts, const Array1<T>& vals) {
  set_data(pts, vals);  
}

template <class T> inline bool Linear3DPWI<T>::get_value(double w, T& res){
  int i;
  if (data_valid && (i=get_interval(w))>=0){
    res=T(X[i].a+X[i].b*w, Y[i].a+Y[i].b*w, Z[i].a+Z[i].b*w);
    return true;
  }  
  else
    return false;
}
	
// takes sorted array of points
template <class T> bool Linear3DPWI<T>::set_data(const Array1<double>& pts, const Array1<T>& vals){
  int sz=0;
  reset();
  if (fill_data(pts) && (sz=points.size())>1){
    X.resize(sz);
    Y.resize(sz);
    Z.resize(sz);
    
    double lb, rb, delta;

    for (int i=0; i<points.size()-1; i++){
      lb=points[i];
      rb=points[i+1];
      delta=rb-lb;
      X[i].a=(vals[i].x()*rb-vals[i+1].x()*lb)/delta;
      X[i].b=(vals[i+1].x()-vals[i].x())/delta;
      Y[i].a=(vals[i].y()*rb-vals[i+1].y()*lb)/delta;
      Y[i].b=(vals[i+1].y()-vals[i].y())/delta;
      Z[i].a=(vals[i].z()*rb-vals[i+1].z()*lb)/delta;
      Z[i].b=(vals[i+1].z()-vals[i].z())/delta;
    }
    return data_valid=true;
  }
  else{
    reset();
    return data_valid=false;
  }
}

} // Math 
} // SCICore

#endif //SCI_LINEARPWI_H__





