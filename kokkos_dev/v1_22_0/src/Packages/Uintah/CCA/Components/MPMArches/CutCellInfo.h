
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
  
   Copyright (C) 2003 University of Utah

KEYWORDS
   cut cells, complex geometry

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <Packages/Uintah/CCA/Components/MPMArches/CutCellInfoP.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Core/Containers/OffsetArray1.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
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
