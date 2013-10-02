/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <CCA/Components/Examples/Interpolator.h>


using namespace Uintah;
using namespace SCIRun;

Interpolator::Interpolator(int /*factor*/)
{ 
    factor_ = 2; 

    refine_support_ = IntVector(0,0,0);
    coarsen_support_ = IntVector(1,1,1);
    max_refine_support_ = 1;
    max_coarsen_support_ = 1;
}


double Interpolator::refine(constNCVariable<double>& variable,
			    IntVector index, Interpolator::PointType /*type*/)
{
    double result;
    if(index[0] & 1) {
	if(index[1] & 1) {
	    if(index[2] & 1) {    // interpolate in all dimensions
	        result = (variable[fineToCoarseIndex(index + IntVector(-1,-1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector(-1,-1, 1))]
	                + variable[fineToCoarseIndex(index + IntVector(-1, 1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector(-1, 1, 1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1,-1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1,-1, 1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 1, 1))]) / 8.0;
	    } else {              // interpolate in 0,1 project in 2
	        result = (variable[fineToCoarseIndex(index + IntVector(-1,-1, 0))]
	                + variable[fineToCoarseIndex(index + IntVector(-1, 1, 0))]
	                + variable[fineToCoarseIndex(index + IntVector( 1,-1, 0))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 1, 0))]) / 4.0;
	    }
	} else {
	    if(index[2] & 1) {    // interpolate in 0,2 project in 1
	        result = (variable[fineToCoarseIndex(index + IntVector(-1, 0,-1))]
	                + variable[fineToCoarseIndex(index + IntVector(-1, 0, 1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 0,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 0, 1))]) / 4.0;
	    } else {              // interpolate in 0 project in 1,2
	        result = (variable[fineToCoarseIndex(index + IntVector(-1, 0, 0))]
	                + variable[fineToCoarseIndex(index + IntVector( 1, 0, 0))]) / 2.0;
	    }
	}
    } else {
	if(index[1] & 1) {
	    if(index[2] & 1) {    // interpolate in 1,2 project in 0
	        result = (variable[fineToCoarseIndex(index + IntVector( 0,-1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 0,-1, 1))]
	                + variable[fineToCoarseIndex(index + IntVector( 0, 1,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 0, 1, 1))]) / 4.0;
	    } else {              // interpolate in 1 project in 0,2
	        result = (variable[fineToCoarseIndex(index + IntVector( 0,-1, 0))]
	                + variable[fineToCoarseIndex(index + IntVector( 0, 1, 0))]) / 2.0;
	    }
	} else {
	    if(index[2] & 1) {    // interpolate in 2 project in 0,1
	        result = (variable[fineToCoarseIndex(index + IntVector( 0, 0,-1))]
	                + variable[fineToCoarseIndex(index + IntVector( 0, 0, 1))]) / 2.0;
	    } else {              // project in all dimensions
	        result = variable[fineToCoarseIndex(index)];
	    }
	}
    }
    
    return result;
}

double Interpolator::refine(constNCVariable<double>& variable1, double weight1,
			    constNCVariable<double>& variable2, double weight2,
			    IntVector index, Interpolator::PointType /*type*/)
{
  double result1, result2;
  if(index[0] & 1) {
    if(index[1] & 1) {
      if(index[2] & 1) {    // interpolate in all dimensions
	result1 = (variable1[fineToCoarseIndex(index + IntVector(-1,-1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector(-1,-1, 1))]
		   + variable1[fineToCoarseIndex(index + IntVector(-1, 1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector(-1, 1, 1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1,-1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1,-1, 1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 1, 1))]) / 8.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector(-1,-1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector(-1,-1, 1))]
		   + variable2[fineToCoarseIndex(index + IntVector(-1, 1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector(-1, 1, 1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1,-1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1,-1, 1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 1, 1))]) / 8.0;
      } else {              // interpolate in 0,1 project in 2
	result1 = (variable1[fineToCoarseIndex(index + IntVector(-1,-1, 0))]
		   + variable1[fineToCoarseIndex(index + IntVector(-1, 1, 0))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1,-1, 0))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 1, 0))]) / 4.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector(-1,-1, 0))]
		   + variable2[fineToCoarseIndex(index + IntVector(-1, 1, 0))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1,-1, 0))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 1, 0))]) / 4.0;
      }
    } else {
      if(index[2] & 1) {    // interpolate in 0,2 project in 1
	result1 = (variable1[fineToCoarseIndex(index + IntVector(-1, 0,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector(-1, 0, 1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 0,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 0, 1))]) / 4.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector(-1, 0,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector(-1, 0, 1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 0,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 0, 1))]) / 4.0;
      } else {              // interpolate in 0 project in 1,2
	result1 = (variable1[fineToCoarseIndex(index + IntVector(-1, 0, 0))]
		   + variable1[fineToCoarseIndex(index + IntVector( 1, 0, 0))]) / 2.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector(-1, 0, 0))]
		   + variable2[fineToCoarseIndex(index + IntVector( 1, 0, 0))]) / 2.0;
      }
    }
  } else {
    if(index[1] & 1) {
      if(index[2] & 1) {    // interpolate in 1,2 project in 0
	result1 = (variable1[fineToCoarseIndex(index + IntVector( 0,-1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 0,-1, 1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 0, 1,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 0, 1, 1))]) / 4.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector( 0,-1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 0,-1, 1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 0, 1,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 0, 1, 1))]) / 4.0;
      } else {              // interpolate in 1 project in 0,2
	result1 = (variable1[fineToCoarseIndex(index + IntVector( 0,-1, 0))]
		   + variable1[fineToCoarseIndex(index + IntVector( 0, 1, 0))]) / 2.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector( 0,-1, 0))]
		   + variable2[fineToCoarseIndex(index + IntVector( 0, 1, 0))]) / 2.0;
	    }
    } else {
      if(index[2] & 1) {    // interpolate in 2 project in 0,1
	result1 = (variable1[fineToCoarseIndex(index + IntVector( 0, 0,-1))]
		   + variable1[fineToCoarseIndex(index + IntVector( 0, 0, 1))]) / 2.0;
	result2 = (variable2[fineToCoarseIndex(index + IntVector( 0, 0,-1))]
		   + variable2[fineToCoarseIndex(index + IntVector( 0, 0, 1))]) / 2.0;
      } else {              // project in all dimensions
	result1 = variable2[fineToCoarseIndex(index)];
	result2 = variable2[fineToCoarseIndex(index)];
      }
    }
  }
  
  return result1*weight1+result2*weight2;
}

double Interpolator::coarsen(const NCVariable<double>& variable, IntVector index, Interpolator::PointType /*type*/)
{
    IntVector fineIndex = coarseToFineIndex(index);

    return ((((variable[fineIndex + IntVector(-1,-1,-1)]
             + variable[fineIndex + IntVector(-1,-1, 1)]
             + variable[fineIndex + IntVector(-1, 1,-1)]
             + variable[fineIndex + IntVector(-1, 1, 1)]
             + variable[fineIndex + IntVector( 1,-1,-1)]
             + variable[fineIndex + IntVector( 1,-1, 1)]
             + variable[fineIndex + IntVector( 1, 1,-1)]
             + variable[fineIndex + IntVector( 1, 1, 1)] ) / 2
            
             + variable[fineIndex + IntVector( 0,-1,-1)]
             + variable[fineIndex + IntVector( 0,-1, 1)]
             + variable[fineIndex + IntVector( 0, 1,-1)]
             + variable[fineIndex + IntVector( 0, 1, 1)]
            
             + variable[fineIndex + IntVector(-1, 0,-1)]
             + variable[fineIndex + IntVector(-1, 0, 1)]
             + variable[fineIndex + IntVector( 1, 0,-1)]
             + variable[fineIndex + IntVector( 1, 0, 1)]
            
             + variable[fineIndex + IntVector(-1,-1, 0)]
             + variable[fineIndex + IntVector(-1, 1, 0)]
             + variable[fineIndex + IntVector( 1,-1, 0)]
             + variable[fineIndex + IntVector( 1, 1, 0)] ) / 2
            
             + variable[fineIndex + IntVector( 0, 0,-1)]
             + variable[fineIndex + IntVector( 0, 0, 1)]
            
             + variable[fineIndex + IntVector( 0,-1, 0)]
             + variable[fineIndex + IntVector( 0, 1, 0)]
            
             + variable[fineIndex + IntVector(-1, 0, 0)]
             + variable[fineIndex + IntVector( 1, 0, 0)] ) / 2
            
             + variable[fineIndex] ) / 8;
}
