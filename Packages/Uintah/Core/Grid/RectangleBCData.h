#ifndef UINTAH_GRID_RectangleBCData_H
#define UINTAH_GRID_RectangleBCData_H

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
   RectangleBCData
   
   
GENERAL INFORMATION

   RectangleBCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   RectangleBCData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class RectangleBCData : public BCGeomBase  {
   public:
     RectangleBCData();
     RectangleBCData(BCData& bc);
     RectangleBCData(const std::string& type);
     RectangleBCData(ProblemSpecP& ps);
     RectangleBCData(Point& low, Point& up);
     virtual ~RectangleBCData();
     RectangleBCData* clone();
     void addBCData(BCData& bc);
     void addBC(BoundCondBase* bc);
     void getBCData(BCData& bc) const;
     bool inside(const Point& p) const;

   private:
     BCData d_bc;
     Point d_min,d_max;
   };

} // End namespace Uintah

#endif




