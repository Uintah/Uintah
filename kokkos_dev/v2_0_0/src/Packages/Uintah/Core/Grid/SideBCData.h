#ifndef UINTAH_GRID_SideBCData_H
#define UINTAH_GRID_SideBCData_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCDataBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;

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

   class SideBCData : public BCDataBase {
   public:
     SideBCData();
     SideBCData(BCData& d_bc);
     SideBCData(const std::string& type);
     SideBCData(ProblemSpecP& ps);
#if 0
     SideBCData(const SideBCData& bc);
     SideBCData& operator=(const SideBCData& bc);
#endif
     virtual ~SideBCData();
     SideBCData* clone();
     void getBCData(BCData& bc) const;
     void addBCData(BCData& bc);
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
     BCData d_bc;
     vector<IntVector> boundary,interior,sfcx,sfcy,sfcz;
   };

} // End namespace Uintah

#endif




