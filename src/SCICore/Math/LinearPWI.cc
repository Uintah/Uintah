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
#include <SCICore/Containers/Array2.h>

// #include <as_default.h>

namespace SCICore{
namespace Math{

using namespace SCICore::Containers;

	LinearPWI::LinearPWI()
	{
	}

	LinearPWI::LinearPWI(Array1<float>& pts, Array1<double>& vals) {
		set_data(pts, vals);
		clean_data(false);
	}
	
	// takes unsorted array of points
	bool LinearPWI::set_data(Array1<float>& pts, Array1<double>& vals){
		if (fill_data(pts, vals) && points.size()>1){
		       //  MSG ("LinearPWI::set_data - inside, filling was OK");
			a.newsize(points.size(), 2);
			for (int i=0; i<points.size()-1; i++){
				a(i, 0)=(values[i]*points[i+1]-values[i+1]*points[i])/(points[i+1]-points[i]);
				a(i, 1)=(values[i+1]-values[i])/(points[i+1]-points[i]);
			}
			return data_valid=true;
		}
		else
		{
		 //  MSG ("LinearPWI::set_data - filling data was not OK, data not valid");
		  return data_valid=false;
		}
	}

} //Math
} //SCICore
