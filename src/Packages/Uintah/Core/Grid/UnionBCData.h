#ifndef UINTAH_GRID_UnionBCData_H
#define UINTAH_GRID_UnionBCData_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;

/**************************************

CLASS
   UnionBCData
   
   
GENERAL INFORMATION

   UnionBCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   UnionBCData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class UnionBCData : public BCGeomBase {
   public:
     UnionBCData();
     UnionBCData(BCData& bc);
     UnionBCData(const UnionBCData& bc);
     UnionBCData& operator=(const UnionBCData& bc);
     UnionBCData(ProblemSpecP& ps);
     virtual ~UnionBCData();
     UnionBCData* clone();
     void getBCData(BCData& bc) const;
     void addBCData(BCData& bc);
     void addBC(BoundCondBase* bc);
     void addBCData(BCGeomBase* bc);
     bool inside(const Point& p) const;
         
   private:
     std::vector<BCGeomBase*> child;
     friend class BCReader;
   };

} // End namespace Uintah

#endif




