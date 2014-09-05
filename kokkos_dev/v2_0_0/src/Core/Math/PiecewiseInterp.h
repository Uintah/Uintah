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
 *  PiecewiseInterp.h: base class for local family of interpolation techniques
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef SCI_PIECEWISEINTERP_H__
#define SCI_PIECEWISEINTERP_H__

#include <Core/Math/MiscMath.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Sort.h>
#include <Core/share/share.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#define msg(m) std::cout << m << std::endl;

namespace SCIRun {


template <class T> class SCICORESHARE PiecewiseInterp {
protected:
  bool data_valid;			// set to true in an interpolation specific cases
  int  curr_intrv;			// number of last interval used
  double min_bnd, max_bnd;                // and its boundaries (for camera path optimizations)
  Array1<double> points;		        // sorted parameter points
  inline bool fill_data(const Array1<double>&);			     
public:
  virtual ~PiecewiseInterp() {};
  inline int get_interval(double);
  virtual bool get_value(double, T&)=0;
  virtual bool set_data(const Array1<double>& pts, const Array1<T>& vals)=0;
  void reset() {
    data_valid=false;
    curr_intrv=-1;
    min_bnd=max_bnd=0;
  };
};

// returns -1 if data is not valid or the value is out of boundaries
template <class T> inline int PiecewiseInterp<T>::get_interval(double w){
  if (data_valid) {
    if (w<min_bnd || w>=max_bnd) {	// taking advantage of smooth parameter changing	
       int lbnd=0, rbnd=points.size()-1, delta;
	  
       if (w>=points[lbnd] && w<=points[rbnd]) {
	 
	 if (curr_intrv >=0)
	   if (w<min_bnd) {			// the series of optimizations that will take advantage
						     // on monotonic parameter changing (camera path interp.)		
	   if (w>=points[curr_intrv-1])
	     lbnd=curr_intrv-1;			
	   rbnd=curr_intrv;
	   
	   } else {
	     if (w<=points[curr_intrv+1])
	       rbnd=curr_intrv+1;
	     lbnd=curr_intrv;
	   }
	 
	 while ((delta=rbnd-lbnd)!=1){
	   if (w<points[lbnd+delta/2])
	     rbnd=lbnd+delta/2;
	   else
	     lbnd+=delta/2;
	 }
	 
	 curr_intrv=lbnd;
	 min_bnd=points[curr_intrv];
	 max_bnd=points[curr_intrv+1];
       }
       else
	 {
	   curr_intrv=(w<points[lbnd])? lbnd:rbnd;
	   min_bnd=0;
	   max_bnd=0;
	 }
    }
  }
  else
    reset();
  
  return curr_intrv;
}

template<class T> inline bool PiecewiseInterp<T>::fill_data(const Array1<double>& pts){
  for (int i=1; i<pts.size(); i++) {
    if ((pts[i]-pts[i-1])<1e-7){
      return false;
    }
  }	

  points.resize(pts.size());
  points=pts;
  msg ("Filled!!!");
  return true;
}

} // End namespace SCIRun

#endif //SCI_INTERPOLATION_H__




