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


#ifndef Uintah_Component_MPMArches_CutCellInfo_h
#define Uintah_Component_MPMArches_CutCellInfo_h

/**************************************

CLASS
   CutCellInfo
   
   Class for storing cut cell information; at present
   it only has a struct that has basic information for
   basic cut cell i/o

GENERAL INFORMATION

   MPMArches.h

   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
KEYWORDS
   cut cells, complex geometry

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <CCA/Components/MPMArches/CutCellInfoP.h>
#include <Core/Util/RefCounted.h>
#include <Core/Containers/OffsetArray1.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/Array3.h>
namespace Uintah {
  using namespace SCIRun;

struct CutCellInfo : public RefCounted {

  int patchindex;
  int tot_cutp;
  int tot_cutu;
  int tot_cutv;
  int tot_cutw;
  int tot_walls;
  int iccst;
  int jccst;
  int kccst;

  CutCellInfo();
  ~CutCellInfo();
};
} // End namespace Uintah

#endif
