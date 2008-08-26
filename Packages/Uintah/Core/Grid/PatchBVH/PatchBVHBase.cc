
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVH.h>
namespace Uintah {
 

  /**************************************

    CLASS
    PatchBVH

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.

    GENERAL INFORMATION

    PatchBVH.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2008 SCI Group

    KEYWORDS
    PatchBVH

    DESCRIPTION
    The PatchBVH is used for querying patches within a given range.
    WARNING

   ****************************************/
  unsigned int PatchBVHBase::leafSize_=4;

} // end namespace Uintah
