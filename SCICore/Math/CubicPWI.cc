/*
 *  LinearPWI.cc: linear piecewise interpolation
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/share/share.h>
#include <SCICore/Math/CubicPWI.h>
#include <SCICore/Containers/Array1.h>
//#include <SCICore/Malloc/Allocator.h>

namespace SCICore{
namespace Math{

using SCICore::Containers::Array1;

CubicPWI::CubicPWI()
{
}

CubicPWI::CubicPWI(const Array1<double>& pts, const Array1<double>& vals) {
  set_data(pts, vals);
}

bool SCICORESHARE first_drv(const Array1<double>& pts, const Array1<double>& vals, Array1<double>& res, int tns){
  const int sz=pts.size();
  cout << "Inside first_der" << endl;
  cout << "Pts array:" << pts << endl;
  cout << "Vals array:" << vals << endl;

  if (sz!=vals.size() || sz<2)
    return false;

  res.remove_all();

  // another way to find tangents
  
  double denom=pts[1]-pts[0];
    
  cout << "About to fill in first point der" << endl;
  if (denom > 10e-10)
    res.add((1-tns)*(vals[1]-vals[0])/denom);
  else 
    return false;
  cout << "About to fill in intermediate points " << endl;
  for (int i=1; i<sz-1; i++){
    if ((denom=pts[i+1]-pts[i-1]) > 10e-10)
      res.add((1-tns)*(vals[i+1]-vals[i-1])/denom);
    else
      return false;
  }
  cout << "About to fill in last point der" << endl;
  denom=pts[sz-1]-pts[sz-2];
  if (denom > 10e-10)
    res.add((1-tns)*(vals[sz-2]-vals[sz-1])/denom);
  else 
    return false;
  
  for (int i=0; i<sz; i++)
    res[i]=0;
  return true;
 
}

// takes sorted array of points
bool CubicPWI::set_data(const Array1<double>& pts, const Array1<double>& vals){
  cout << "About to fill in data in set_data (1D)!!!" << endl;
  cout << "pts-array:" << endl;
  cout << pts;

  cout << "vals-array:" << endl;
  cout << pts;

  reset();
  int sz;

  if (fill_data(pts) && (sz=points.size())>1 && sz==vals.size()){
    cout << "Inside set_data!!!" << endl;

    p.resize(sz);
    Array1<double> ders;
    if (first_drv(points, vals, ders, 0)){

      cout << "First Derivatives are done!!! Here they are:" << endl;
      cout << "x-dir:" << ders << endl;

      data_valid=true;
      double a, b, c, d, delta, tmp;
     
      for (int i=0; i<sz-1; i++)
	if ( (delta=points[i+1]-points[i]) >10e-9){
	  a=vals[i];
	  b=ders[i];
	  c=(3*(vals[i+1]-vals[i])/delta-2*ders[i]-ders[i+1])/delta;
	  d=(ders[i]+ders[i+1]-2*(vals[i+1]-vals[i])/delta)/(delta*delta);
	
	  tmp=points[i];
	  p[i].a=a+tmp*(tmp*(c-tmp*d)-b);
	  p[i].b=b-tmp*(2*c-3*tmp*d);
	  p[i].c=c-3*tmp*d;
	  p[i].d=d;

	  cout << "Interval: " << points[i] << ", " << points[i+1] << endl;
	  cout << "Coeff. are: " << p[i].a << endl << p[i].b << endl << p[i].c << endl << p[i].d << endl;
	  
	}
	else {
	  reset();
	  break;
	}
    }
  }

  return data_valid;
}

} //Math
} //SCICore



