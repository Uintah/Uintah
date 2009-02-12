/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef Math_Spline_h
#define Math_Spline_h 1

namespace rtrt {

inline double CRSpline(double t, double p1, double p2,
		double p3, double p4)
{
	double t1=((-1*t+2)*t-1)*t;
	double t2=((3*t-5)*t+0)*t+2;
	double t3=((-3*t+4)*t+1)*t;
	double t4=((1*t-1)*t+0)*t;
	return (p1*t1+p2*t2+p3*t3+p4*t4)*0.5;
}

inline double dCRSpline(double t, double p1, double p2,
	double p3, double p4)
{
	double t1=(-3*t+4)*t-1;
	double t2=(9*t-10)*t;
	double t3=(-9*t+8)*t+1;
	double t4=(3*t-2)*t;
	return (p1*t1+p2*t2+p3*t3+p4*t4)*0.5;
}

} // end namespace rtrt

#endif
