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

#include <SCICore/Math/LinearPWI.h>
#include <SCICore/share/share.h>
#include <SCICore/Containers/Sort.h>
#include <SCICore/Containers/Array1.h>

namespace SCICore{
namespace Math{

using namespace SCICore::Containers;

LinearPWI::LinearPWI()
{
}

LinearPWI::LinearPWI(const Array1<double>& pts, const Array1<double>& vals) {
  set_data(pts, vals);
}

// takes sorted array of points
bool LinearPWI::set_data(const Array1<double>& pts, const Array1<double>& vals){
  reset();
  if (fill_data(pts) && points.size()>1){
    p.resize(points.size());
    for (int i=0; i<points.size()-1; i++){
      p[i].a=(vals[i]*points[i+1]-vals[i+1]*points[i])/(points[i+1]-points[i]);
      p[i].b=(vals[i+1]-vals[i])/(points[i+1]-points[i]);
    }
    return data_valid=true;
  }
  else
    {
      return data_valid=false;
    }
}

} //Math
} //SCICore
