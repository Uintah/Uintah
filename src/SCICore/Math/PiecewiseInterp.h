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

#include <SCICore/Math/Interpolation.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Sort.h>
#include <SCICore/share/share.h>
/* #include <as_debug.h> */

namespace SCICore{
namespace Math{

using namespace SCICore::Containers;

class SCICORESHARE PiecewiseInterp: public Interpolation {
protected:
	bool data_valid;			// set to true in an interpolation specific cases
	int  curr_intrv;			// number of last interval used
        double min_bnd, max_bnd;                // and its boundaries (for camera path optimizations)
	Array1<float> points;		// sorted parameter points
	Array1<double> values;		// corresponding values; not gets populated in certain types of interp.
	inline bool fill_data(Array1<float>&, Array1<double>&);
	PiecewiseInterp();
public:
	virtual double get_value(double)=0;
	virtual ~PiecewiseInterp() {};
	inline int get_interval(double);
	void reset() {data_valid=false;};
};

// returns -1 if data is not valid or the value is out of boundaries
inline int PiecewiseInterp::get_interval(double w){
	if (data_valid) {
	       /*  MSG ("Getting into get_interval() with w=:"); */
/* 		MSG (w); */
		if (w<min_bnd || w>=max_bnd) {	// taking advantage of smooth parameter changing	
			int lbnd=0, rbnd=points.size()-1, delta;

			if (w>=points[lbnd] && w<=points[rbnd]) {
		       
			  /*		if (w<min_bnd) {			// the series of optimizations that will take advantage
										// on monotonic parameter changing (camera path interp.)		
					if (w>=points[curr_intrv-1])
						lbnd=curr_intrv-1;			
					rbnd=curr_intrv;
				} else {
					if (w<=points[curr_intrv+1])
						rbnd=curr_intrv+1;
					lbnd=curr_intrv;
				}
				*/
				while ((delta=rbnd-lbnd)!=1)
				{
					if (w<points[lbnd+delta/2])
						rbnd=lbnd+delta/2;
					else
						lbnd+=delta/2;
				}

				curr_intrv=lbnd;
				/* MSG ("Interval found:"); */
/* 				MSG (curr_intrv); */
				min_bnd=points[curr_intrv];
				max_bnd=points[curr_intrv+1];
			}
			else
			{
				curr_intrv=-1;
				min_bnd=0;
				max_bnd=0;
			}
		}
	}
	else
	{
		// no valid data
		curr_intrv=-1;
		min_bnd=0;
		max_bnd=0;
	}
	/* MSG ("PiecewiseInterp::get_interval() - Returning Interval:"); */
/* 	MSG (curr_intrv); */
	return curr_intrv;
}

inline bool PiecewiseInterp::fill_data(Array1<float>& pts, Array1<double>& vals){
		if (pts.size()!=vals.size() || pts.size()==0) {
			return false;
		}
		Array1<unsigned int> ind;
		SortObjs sorter;
		
		/* MSG ("PiecewiseInterp::fill_data() - pts[] before sorting"); */
/* 		MSG (pts); */

		sorter.DoQSort(pts, ind);

		/* MSG ("PiecewiseInterp::fill_data() - pts[] after sorting"); */
/* 		MSG (pts); */
		
/* 		MSG ("PiecewiseInterp::fill_data() - ind[] after filling by DoQSort()"); */
/* 		MSG (ind); */

		points.resize(pts.size());
		values.resize(vals.size());

		unsigned int index;
		
		for (int i=0; i<pts.size(); i++) {
			index=ind[i];
			points[i]=pts[index];
			values[i]=vals[index];
		}
		
		/* MSG ("PiecewiseInterp::fill_data() - points[] after filling"); */
/* 		MSG (points); */
/* 		MSG ("PiecewiseInterp::fill_data() - points[] after filling"); */
/* 		MSG (values); */
		
		for (int i=points.size()-1; i>=1; i--) {
			if (Abs(points[i]-points[i-1])<10e-5) // some possible data cleaning or bad data return
				if (Abs(values[i]-values[i-1])>10e-5 || !cldata) 		
					return data_valid=false;
				else 
				{
					points.remove(i);
					values.remove(i);
				}
			       
		}
		return true;
}

} // Math
} // SCICore

#endif //SCI_INTERPOLATION_H__




