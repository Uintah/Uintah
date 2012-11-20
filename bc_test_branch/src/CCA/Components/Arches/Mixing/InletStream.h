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

//----- InletStream.h -----------------------------------------------

#ifndef Uintah_Components_Arches_InletStream_h
#define Uintah_Components_Arches_InletStream_h

/**************************************
CLASS
   InletStream
   
   Class InletStream creates and stores the mixing variables that are used in Arches

GENERAL INFORMATION
   InletStream.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   July 20, 2000
   
   C-SAFE 
   

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/
#include <Core/Geometry/IntVector.h>

#include <vector>

namespace Uintah {
  using namespace SCIRun;
    class InletStream {
    public:
      InletStream();
      InletStream(int numMixVars, int numMixStatVars, int numRxnVars);
      ~InletStream();
      std::vector<double> d_mixVars;
      std::vector<double> d_mixVarVariance;
      double d_f2; 
      bool d_has_second_mixfrac; 
      double d_enthalpy;
      bool d_initEnthalpy;
      std::vector<double> d_rxnVars;
      int d_axialLoc;
      double d_scalarDisp;
      double cellvolume;
      double d_heatloss; 
    
      IntVector d_currentCell;

    }; // End class InletStream

}  // End namespace Uintah

#endif

