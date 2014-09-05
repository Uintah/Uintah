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
#include <Core/Grid/Variables/Utils.h>

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <math.h>

using namespace std;

#define d_SMALLNUM 1e-100

namespace Uintah {
//______________________________________________________________________
//   This function examines all the values for being positive.  If a 
//   negative value or nan is found the function returns false along 
//   with the first cell index.
template< class T >
bool
areAllValuesPositive( T & src, IntVector & neg_cell )
{ 
  double    numCells = 0;
  double    sum_src = 0;
  int       sumNan = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  CellIterator iterLim = CellIterator(l,h);
  
  for(CellIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumNan += isnan(src[c]);       // check for nans
    sum_src += src[c]/(fabs(src[c]) + d_SMALLNUM);
    numCells++;
  }

  // now find the first cell where the value is < 0   
  if ( (fabs(sum_src - numCells) > 1.0e-2) || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (src[c] < 0.0 || isnan(src[c]) !=0) {
        neg_cell = c;
        return false;
      }
    }
  } 
  neg_cell = IntVector(0,0,0); 
  return true;      
}

//______________________________________________________________________
//   This function examines all the values for being numbers.  If a 
//   inf or nan is found the function returns false along 
//   with the first cell index.
template< class T >
bool
areAllValuesNumbers( T & src, IntVector & neg_cell )
{ 
  double    numCells = 0;
  double    sum_src = 0;
  int       sumNan = 0;
  int       sumInf = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  CellIterator iterLim = CellIterator(l,h);
  
  for(CellIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumInf += isinf(src[c]);       // check for infs
    sumNan += isnan(src[c]);       // check for nans
    sum_src += src[c]/(fabs(src[c]) + d_SMALLNUM);
    numCells++;
  }

  // now find the first cell where the value is inf or nan   
  if ( (fabs(sum_src - numCells) > 1.0e-2) || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (isinf(src[c]) != 0 || isnan(src[c]) !=0) {
        neg_cell = c;
        return false;
      }
    }
  } 
  neg_cell = IntVector(0,0,0); 
  return true;      
}

// Explicit template instantiations:
template bool areAllValuesPositive( CCVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCXVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCYVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCZVariable<double> &, IntVector & );

// Explicit template instantiations:
template bool areAllValuesNumbers( CCVariable<double> &, IntVector & );
template bool areAllValuesNumbers( SFCXVariable<double> &, IntVector & );
template bool areAllValuesNumbers( SFCYVariable<double> &, IntVector & );
template bool areAllValuesNumbers( SFCZVariable<double> &, IntVector & );

} // end namespace Uintah
