#ifndef UINTAH_GRID_UnionBCData_H
#define UINTAH_GRID_UnionBCData_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCDataBase.h>
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

   class UnionBCData : public BCDataBase {
   public:
     UnionBCData();
     UnionBCData(BCData& bc);
     UnionBCData(const std::string& type);
     UnionBCData(const UnionBCData& bc);
     UnionBCData& operator=(const UnionBCData& bc);
     UnionBCData(ProblemSpecP& ps);
     virtual ~UnionBCData();
     UnionBCData* clone();
     void getBCData(BCData& bc) const;
     void addBCData(BCData& bc);
     void addBCData(BCDataBase* bc);
     void setBoundaryIterator(std::vector<IntVector>& b);
     void setInteriorIterator(std::vector<IntVector>& i);
     void setSFCXIterator(std::vector<IntVector>& i);
     void setSFCYIterator(std::vector<IntVector>& i);
     void setSFCZIterator(std::vector<IntVector>& i);
     void getBoundaryIterator(std::vector<IntVector>& b) const;
     void getInteriorIterator(std::vector<IntVector>& i) const;
     void getSFCXIterator(std::vector<IntVector>& i) const;
     void getSFCYIterator(std::vector<IntVector>& i) const;
     void getSFCZIterator(std::vector<IntVector>& i) const;
     bool inside(const Point& p) const;
         
   private:
     std::vector<BCDataBase*> child;
     std::vector<IntVector> boundary,interior,sfcx,sfcy,sfcz;
   };

} // End namespace Uintah

#endif




