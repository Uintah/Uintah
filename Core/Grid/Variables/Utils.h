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
#ifndef Packages_Uintah_Core_Grid_Variables_Utils_h
#define Packages_Uintah_Core_Grid_Variables_Utils_h

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {

//______________________________________________________________________
//   This function examines all the values for being positive.  If a 
//   negative value or nan is found the function returns false along 
//   with the first cell index.
template< class T >
bool areAllValuesPositive( T                 & src, 
                           SCIRun::IntVector & neg_cell );

//______________________________________________________________________
//   This function examines all the values for being numbers.  If a 
//   inf or nan is found the function returns false along 
//   with the first cell index.
template< class T >
bool areAllValuesNumbers( T                 & src, 
                           SCIRun::IntVector & neg_cell );

bool areAllNodeValuesNumbers( NCVariable<Vector>  & src, 
                           SCIRun::IntVector & neg_cell );
}

#endif
