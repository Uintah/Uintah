
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
  
   Copyright (C) 2000 University of Utah

KEYWORDS
   cut cells, complex geometry

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

#include <Packages/Uintah/CCA/Components/MPMArches/CutCellInfoP.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Core/Containers/OffsetArray1.h>
namespace Uintah {
  using namespace SCIRun;

struct CutCellInfo : public RefCounted {

  int nwalls;
  int iwst;
  int jwst;
  int kwst;
  int nccells;
  int iccst;
  int jccst;
  int kccst;

  CutCellInfo();
  ~CutCellInfo();
};
} // End namespace Uintah

#endif
