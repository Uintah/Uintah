/*
 *  LinearPWI.h: linear piecewise interpolation
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
#include <SCICore/Containers/Array2.h>
/* #include <as_default.h> */

namespace SCICore {
namespace Math {

using namespace SCICore::Containers;

class SCICORESHARE LinearPWI: public PiecewiseInterp {
public:
	LinearPWI();
	LinearPWI(Array1<float>&, Array1<double>&);

	bool set_data(Array1<float>&, Array1<double>&);
	inline double get_value(double);

private:
	Array2<double> a;
};

inline double LinearPWI::get_value(double w){
	int interv;
	if (data_valid && (interv=get_interval(w))>=0)
		return a(interv, 0)+a(interv, 1)*w;
	else
	  {
	   /*  MSG("LinearPWI::get_value() - cann't obtain data"); */
/* 	    MSG(data_valid); */
/* 	    MSG(interv); */
		return 0; // need to throw some exception !!!
			       }
}

} // Math 
} // SCICore

#endif //SCI_LINEARPWI_H__
