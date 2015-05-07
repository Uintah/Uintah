/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#ifndef Uintah_Component_Arches_CellInformation_h
#define Uintah_Component_Arches_CellInformation_h

/**************************************

CLASS
   CellInformation
   
   Short description...

GENERAL INFORMATION

   Arches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   grid, nonuniform grid

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <CCA/Components/Arches/CellInformationP.h>
#include <Core/Util/RefCounted.h>
#include <Core/Containers/OffsetArray1.h>
namespace Uintah {
  class Patch;

struct CellInformation : public RefCounted {
  // for non-uniform grid these values will come from the
  // patch but for the time being we're assigning the values 
  // locally
  SCIRun::OffsetArray1<double> xx;
  SCIRun::OffsetArray1<double> yy;
  SCIRun::OffsetArray1<double> zz;
  // p-cell geom information
  // x- direction
  SCIRun::OffsetArray1<double> dxep;
  SCIRun::OffsetArray1<double> dxpw;
  SCIRun::OffsetArray1<double> sew;
  // y- direction
  SCIRun::OffsetArray1<double> dynp;
  SCIRun::OffsetArray1<double> dyps;
  SCIRun::OffsetArray1<double> sns;
  // z-direction
  SCIRun::OffsetArray1<double> dztp;
  SCIRun::OffsetArray1<double> dzpb;
  SCIRun::OffsetArray1<double> stb;
  //u-cell geom info
  SCIRun::OffsetArray1<double> xu;
  SCIRun::OffsetArray1<double> dxepu;
  SCIRun::OffsetArray1<double> dxpwu;
  SCIRun::OffsetArray1<double> sewu;
  //v-cell geom info
  SCIRun::OffsetArray1<double> yv;
  SCIRun::OffsetArray1<double> dynpv;
  SCIRun::OffsetArray1<double> dypsv;
  SCIRun::OffsetArray1<double> snsv;
  //w-cell geom info
  SCIRun::OffsetArray1<double> zw;
  SCIRun::OffsetArray1<double> dztpw;
  SCIRun::OffsetArray1<double> dzpbw;
  SCIRun::OffsetArray1<double> stbw;
  //differencing factors for p-cell
  // x-direction
  SCIRun::OffsetArray1<double> cee;
  SCIRun::OffsetArray1<double> cww;
  SCIRun::OffsetArray1<double> cwe;
  SCIRun::OffsetArray1<double> ceeu;
  SCIRun::OffsetArray1<double> cwwu;
  SCIRun::OffsetArray1<double> cweu;
  SCIRun::OffsetArray1<double> efac;
  SCIRun::OffsetArray1<double> wfac;
  // y-direction
  SCIRun::OffsetArray1<double> cnn;
  SCIRun::OffsetArray1<double> css;
  SCIRun::OffsetArray1<double> csn;
  SCIRun::OffsetArray1<double> cnnv;
  SCIRun::OffsetArray1<double> cssv;
  SCIRun::OffsetArray1<double> csnv;
  SCIRun::OffsetArray1<double> nfac;
  SCIRun::OffsetArray1<double> sfac;
  // z-direction
  SCIRun::OffsetArray1<double> ctt;
  SCIRun::OffsetArray1<double> cbb;
  SCIRun::OffsetArray1<double> cbt;
  SCIRun::OffsetArray1<double> cttw;
  SCIRun::OffsetArray1<double> cbbw;
  SCIRun::OffsetArray1<double> cbtw;
  SCIRun::OffsetArray1<double> tfac;
  SCIRun::OffsetArray1<double> bfac;
  // factors for differencing u-cell
  SCIRun::OffsetArray1<double> fac1u;
  SCIRun::OffsetArray1<double> fac2u;
  SCIRun::OffsetArray1<int> iesdu;
  SCIRun::OffsetArray1<double> fac3u;
  SCIRun::OffsetArray1<double> fac4u;
  SCIRun::OffsetArray1<int> iwsdu;
  // factors for differencing v-cell
  SCIRun::OffsetArray1<double> fac1v;
  SCIRun::OffsetArray1<double> fac2v;
  SCIRun::OffsetArray1<int> jnsdv;
  SCIRun::OffsetArray1<double> fac3v;
  SCIRun::OffsetArray1<double> fac4v;
  SCIRun::OffsetArray1<int> jssdv;
  // factors for differencing w-cell
  SCIRun::OffsetArray1<double> fac1w;
  SCIRun::OffsetArray1<double> fac2w;
  SCIRun::OffsetArray1<int> ktsdw;
  SCIRun::OffsetArray1<double> fac3w;
  SCIRun::OffsetArray1<double> fac4w;
  SCIRun::OffsetArray1<int> kbsdw;
  // CC variable interpolation factors
  SCIRun::OffsetArray1<double> fac1ew;
  SCIRun::OffsetArray1<double> fac2ew;
  SCIRun::OffsetArray1<int> e_shift;
  SCIRun::OffsetArray1<double> fac3ew;
  SCIRun::OffsetArray1<double> fac4ew;
  SCIRun::OffsetArray1<int> w_shift;
  SCIRun::OffsetArray1<double> fac1ns;
  SCIRun::OffsetArray1<double> fac2ns;
  SCIRun::OffsetArray1<int> n_shift;
  SCIRun::OffsetArray1<double> fac3ns;
  SCIRun::OffsetArray1<double> fac4ns;
  SCIRun::OffsetArray1<int> s_shift;
  SCIRun::OffsetArray1<double> fac1tb;
  SCIRun::OffsetArray1<double> fac2tb;
  SCIRun::OffsetArray1<int> t_shift;
  SCIRun::OffsetArray1<double> fac3tb;
  SCIRun::OffsetArray1<double> fac4tb;
  SCIRun::OffsetArray1<int> b_shift;
  // constructor computes the values
  CellInformation(const Patch*);
  ~CellInformation();
};
} // End namespace Uintah



#endif

