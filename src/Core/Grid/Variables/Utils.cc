/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
 */#include <Core/Grid/Variables/Utils.h>

#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <cmath>

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
    sumNan += std::isnan(src[c]);       // check for nans
    sum_src += src[c]/(fabs(src[c]) + d_SMALLNUM);
    numCells++;
  }

  // now find the first cell where the value is < 0   
  if ( (fabs(sum_src - numCells) > 1.0e-2) || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (src[c] < 0.0 || std::isnan(src[c]) !=0) {
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
is_NanInf( T & src, IntVector & badCell )
{ 
  int  sumNan = 0;
  int  sumInf = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  CellIterator iterLim = CellIterator(l,h);
  
  for(CellIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumInf += std::isinf(src[c]);       // check for infs
    sumNan += std::isnan(src[c]);       // check for nans
  }

  // now find the first cell where the value is inf or nan   
  if ( sumInf !=0 || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (std::isinf(src[c]) != 0 || std::isnan(src[c]) !=0) {
        badCell = c;
        return true;
      }
    }
  } 
  badCell = IntVector(0,0,0); 
  return false;      
}

//______________________________________________________________________
//   This method examines all the values and looks for an Inf or Nan.
//   if found it returns true along with the first cell index.
template< class T >
bool
is_NanInf_V( T & src, IntVector & badCell )
{ 
  int       sumNan = 0;
  int       sumInf = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  CellIterator iterLim = CellIterator(l,h);
  
  for(CellIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumInf += src[c].Vector::isinf();       // check for infs
    sumNan += src[c].Vector::isnan();       // check for nans
  }

  // now find the first cell where the value is inf or nan   
  if ( sumInf !=0 || sumNan !=0) {
    for(CellIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (src[c].Vector::isinf() == true || src[c].Vector::isnan() == true) {
        badCell = c;
        return true;
      }
    }
  } 
  badCell = IntVector(0,0,0); 
  return false;      
}

//______________________________________________________________________
//
//template< class T >
bool
is_NanInf(NCVariable<Vector> & src, IntVector & neg_node )
{
  double    numCells = 0;
  double    sum_src = 0;
  int       sumNan = 0;
  int       sumInf = 0;
  IntVector l = src.getLowIndex();
  IntVector h = src.getHighIndex();
  NodeIterator iterLim = NodeIterator(l,h);

  for(NodeIterator iter=iterLim; !iter.done();iter++) {
    IntVector c = *iter;
    sumInf += std::isinf(src[c].length());       // check for infs
    sumNan += std::isnan(src[c].length());       // check for nans
    sum_src += src[c].length()/(fabs(src[c].length()) + d_SMALLNUM);
    numCells++;
  }

  // now find the first cell where the value is inf or nan
  if ( (fabs(sum_src - numCells) > 1.0e-2) || sumNan !=0) {
    for(NodeIterator iter=iterLim; !iter.done();iter++) {
      IntVector c = *iter;
      if (std::isinf(src[c].length()) != 0 || std::isnan(src[c].length()) !=0) {
        neg_node = c;
        return false;
      }
    }
  }
  neg_node = IntVector(0,0,0);
  return true;
}

//______________________________________________________________________
//
// Explicit template instantiations:
template bool areAllValuesPositive( CCVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCXVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCYVariable<double> &, IntVector & );
template bool areAllValuesPositive( SFCZVariable<double> &, IntVector & );
template bool areAllValuesPositive( constCCVariable<double> &, IntVector & );
template bool areAllValuesPositive( constSFCXVariable<double> &, IntVector & );
template bool areAllValuesPositive( constSFCYVariable<double> &, IntVector & );
template bool areAllValuesPositive( constSFCZVariable<double> &, IntVector & );

// Explicit template instantiations:
template bool is_NanInf( CCVariable<double> &, IntVector & );
template bool is_NanInf( SFCXVariable<double> &, IntVector & );
template bool is_NanInf( SFCYVariable<double> &, IntVector & );
template bool is_NanInf( SFCZVariable<double> &, IntVector & );
template bool is_NanInf( NCVariable<double> &, IntVector & );
template bool is_NanInf( constCCVariable<double> &, IntVector & );
template bool is_NanInf( constSFCXVariable<double> &, IntVector & );
template bool is_NanInf( constSFCYVariable<double> &, IntVector & );
template bool is_NanInf( constSFCZVariable<double> &, IntVector & );
template bool is_NanInf( constNCVariable<double> &, IntVector & );

template bool is_NanInf_V( CCVariable<Vector> &, IntVector & );
template bool is_NanInf_V( SFCXVariable<Vector> &, IntVector & );
template bool is_NanInf_V( SFCYVariable<Vector> &, IntVector & );
template bool is_NanInf_V( SFCZVariable<Vector> &, IntVector & );
template bool is_NanInf_V( NCVariable<Vector> &, IntVector & );

template bool is_NanInf_V( constCCVariable<Vector> &, IntVector & );
template bool is_NanInf_V( constSFCXVariable<Vector> &, IntVector & );
template bool is_NanInf_V( constSFCYVariable<Vector> &, IntVector & );
template bool is_NanInf_V( constSFCZVariable<Vector> &, IntVector & );
template bool is_NanInf_V( constNCVariable<Vector> &, IntVector & );


//template bool areAllNodeValuesNumbers( NCVariable<double> &, IntVector & );
} // end namespace Uintah
