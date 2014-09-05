/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

#include <Core/Math/LinearPWI.h>
#include <Core/share/share.h>
#include <Core/Containers/Sort.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {


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

} // End namespace SCIRun
