#ifndef UINTAH_GRID_DifferenceBCData_H
#define UINTAH_GRID_DifferenceBCData_H

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
   DifferenceBCData
   
   
GENERAL INFORMATION

   DifferenceBCData.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   DifferenceBCData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class DifferenceBCData : public BCGeomBase  {
   public:
     DifferenceBCData();
     DifferenceBCData(const DifferenceBCData& rhs);
     DifferenceBCData& operator=(const DifferenceBCData& bc);
     DifferenceBCData(BCGeomBase* p1,BCGeomBase* p2);
     virtual ~DifferenceBCData();
     DifferenceBCData* clone();
     void getBCData(BCData& bc) const;
     void addBCData(BCData& bc);
     void addBC(BoundCondBase* bc);
     bool inside(const Point& p) const;

   private:
     BCGeomBase* left;
     BCGeomBase* right;

     friend class BCReader;
   };

} // End namespace Uintah

#endif




