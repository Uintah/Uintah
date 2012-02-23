/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


/*
*
* Template Numerical Toolkit (TNT): Linear Algebra Module
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/


#ifndef TNT_H
#define TNT_H



//---------------------------------------------------------------------
// Define this macro if you want  TNT to track some of the out-of-bounds
// indexing. This can encur a small run-time overhead, but is recommended 
// while developing code.  It can be turned off for production runs.
// 
//       #define TNT_BOUNDS_CHECK
//---------------------------------------------------------------------
//

//#define TNT_BOUNDS_CHECK



#include "tnt_version.h"
/*#include "tnt_math_utils.h"*/
#include "tnt_array1d.h"
#include "tnt_array2d.h"
#include "tnt_array3d.h"
#include "tnt_array1d_utils.h"
#include "tnt_array2d_utils.h"
#include "tnt_array3d_utils.h"

#include "tnt_fortran_array1d.h"
#include "tnt_fortran_array2d.h"
#include "tnt_fortran_array3d.h"
#include "tnt_fortran_array1d_utils.h"
#include "tnt_fortran_array2d_utils.h"
#include "tnt_fortran_array3d_utils.h"

//#include "tnt_sparse_matrix_csr.h"

/*#include "tnt_stopwatch.h"*/
#include "tnt_subscript.h"
#include "tnt_vec.h"
#include "tnt_cmat.h"


#endif
// TNT_H
