#ifndef UINTAH_GRID_SideBCData_H
#define UINTAH_GRID_SideBCData_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;
 using std::vector;

/**************************************

CLASS
   SideBCData
   
   
GENERAL INFORMATION

   SideBCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SideBCData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SideBCData : public BCGeomBase {
   public:
     SideBCData();
     SideBCData(BCData& d_bc);
#if 0
     SideBCData(const SideBCData& bc);
     SideBCData& operator=(const SideBCData& bc);
#endif
     virtual ~SideBCData();
     SideBCData* clone();
     void getBCData(BCData& bc) const;
     void addBCData(BCData& bc);
     void addBC(BoundCondBase* bc);
     bool inside(const Point& p) const;
     
   private:
     BCData d_bc;
   };

} // End namespace Uintah

#endif




