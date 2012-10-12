/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef MakeTableFunction_H
#define MakeTableFunction_H

class MakeTableFunction{
public:

  void fourArrayTable(const int  &arraySize,
		      const double *array1,
		      const double *array2,
		      const double *array3,
		      const double *array4,
		      char *_argv);

  void twoArrayTable(const int  &arraySize,
		     const double *array1,
		     const double *array2,
		     char *_argv);

  
  void threeArrayTable(const int  &arraySize,
		       const double *array1,
		       const double *array2,
		       const double *array3,
		       char *_argv);

  void coupleTwoArrayTable(const int &arraySize,
			   const double *array1,
			   const double *array2,
			   const int &changelineNo,
			   char *_argv);

  
  // use const whenever possible, const double *p -- non const pointer, const data
  
  


  // not ready for use yet, need to figure out how to deal with real surfaces index
  
  void CylinderVolTableMake(const double *X, const double *Y, const double *Z,
			    const double &R, // R is the radius of the cylinder
			    const double &circle_x, const double &circle_y,
			    const int &xno, const int &yno, const int &zno,
			    int  TopStartNo,
			    int  BottomStartNo,
			    int  FrontStartNo,
			    int  BackStartNo,
			    int  LeftStartNo,
			    int  RightStartNo,
			    int  TopBottomNo,
			    int  FrontBackNo,
			    int  LeftRightNo,
			    char * _argv);

  
 

  // properties of vol-- T, kl, scatter, emiss
  void VolProperty(  const int  &VolElementNo,
		     const double *T_Vol,
		     const double *kl_Vol,
		     const double *scatter_Vol,
		     const double *emiss_Vol,
		     char *_argv);

  void VolProperty_Array(const int &VolElementNo,
			 const double *T_Vol,
			 const double *kl_Vol,
			 const double *scatter_Vol,
			 const double *emiss_Vol,
			 double *VolArray);
  
  // properties of real surface-- T, alpha, rs, rd, emiss
  void RealSurfaceProperty(  int  TopStartNo,
			     int  sumElementNo,
			     const double *T_surface,
			     const double *absorb_surface,
			     const double *rs_surface,
			     const double *rd_surface,
			     const double *emiss_surface,
			     char *_argv);

  
  void RealSurfaceProperty_Array(int TopStartNo,
				 int surfaceElementNo,
				 const double *T_surface,
				 const double *absorb_surface,
				 const double *rs_surface,
				 const double *rd_surface,
				 const double *emiss_surface,
				 double *SurfaceArray);

  
  void singleArrayTable(const double *Array, const int &ArraySize,
			const int &No, char * _argv);
  
  void singleIntArrayTable(const int *Array,
			   const int &ArraySize,
			   const int &No,
			   char *_argv);
  
  void q_Q(const double *q_surface, const double *qdiv,
	   const int &VolElementNo, const int &totalElementNo, char *_argv);

  void vtkSurfaceTableMake( const char *_argv, const int &xnop,
			    const int &ynop, const int &znop,
			    const double *X, const double *Y,
			    const double *Z,
			    const int &surfaceElementNo,
			    const double *q_surface,
			    const double *Q_surface);
  

  void vtkVolTableMake(const char *_argv, const int &xnop,
		       const int &ynop, const int &znop,
		       const double *X, const double *Y,
		       const double *Z,
		       const int &VolElementNo,
		       const double *qdiv,
		       const double *Qdiv);

  void q_top(const double *Array,
	     const int &xno, const int &yno,
	     char *_argv);

  void q_bottom(const double *Array,
		const int &xno, const int &yno,
		char *_argv);
  
  
};

#endif
