#ifndef UINTAH_GRID_CircleBCData_H
#define UINTAH_GRID_CircleBCData_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;

/**************************************

CLASS
   CircleBCData
   
   
GENERAL INFORMATION

   CircleBCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CircleBCData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CircleBCData : public BCGeomBase  {
   public:
     CircleBCData();
     CircleBCData(BCData& bc);
     CircleBCData(Point& p, double radius);
     virtual ~CircleBCData();
     CircleBCData* clone();
     void addBCData(BCData& bc);
     void addBC(BoundCondBase* bc);
     void getBCData(BCData& bc) const;
     bool inside(const Point& p) const;
         
   private:
     BCData d_bc;
     double d_radius;
     Point  d_origin;
   };

} // End namespace Uintah

#endif




