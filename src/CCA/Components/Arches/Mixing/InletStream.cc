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


#include <CCA/Components/Arches/Mixing/InletStream.h>

using namespace Uintah;
using namespace std;

InletStream::InletStream()
{
  d_axialLoc = 0;
  d_enthalpy = 0;
  //d_scalarDisp = 0.0;

  d_currentCell = IntVector (-2, -2, -2); 
  d_f2 = 0.;
}

InletStream::InletStream(int numMixVars, int numMixStatVars, int numRxnVars)
{
  d_mixVars = vector<double>(numMixVars);
  d_mixVarVariance = vector<double>(numMixStatVars);
  d_rxnVars = vector<double>(numRxnVars);
  d_axialLoc = 0;
  d_enthalpy = 0;
  d_initEnthalpy = false;
  d_f2 = 0.;
  
  /* For a grid [0,n] border cells are -1 and n+1.
   * Since -1 is taken by a border cell, we must use -2
   * to designate an invalid cell. */
  d_currentCell = IntVector (-2, -2, -2);
  
  //???What about enthalpy???
}

InletStream::~InletStream()
{
}

//
//
