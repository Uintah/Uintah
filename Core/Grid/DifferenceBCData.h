#ifndef UINTAH_GRID_DifferenceBCData_H
#define UINTAH_GRID_DifferenceBCData_H

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

   class DifferenceBCData : public BCDataBase  {
   public:
     DifferenceBCData();
     DifferenceBCData(const std::string& type);
     DifferenceBCData(ProblemSpecP& ps);
     DifferenceBCData(const DifferenceBCData& rhs);
     DifferenceBCData& operator=(const DifferenceBCData& bc);
     DifferenceBCData(BCDataBase* p1,BCDataBase* p2);
     virtual ~DifferenceBCData();
     DifferenceBCData* clone();
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
     BCDataBase* left;
     BCDataBase* right;

     std::vector<IntVector> boundary,interior,sfcx,sfcy,sfcz;
   };

} // End namespace Uintah

#endif




