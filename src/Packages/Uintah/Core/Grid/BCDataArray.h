#ifndef UINTAH_GRID_BCDataArray_H
#define UINTAH_GRID_BCDataArray_H

#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCDataBase.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

/**************************************

CLASS
   BCDataArray
   
   
GENERAL INFORMATION

   BCDataArray.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BCDataArray

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class BCDataArray {
   public:
     BCDataArray();
     BCDataArray(BCData& bc);
     BCDataArray(const std::string& type);
     BCDataArray(const BCDataArray& bc);
     BCDataArray& operator=(const BCDataArray& bc);
     BCDataArray(ProblemSpecP& ps);
     ~BCDataArray();
     BCDataArray* clone();
     void getBCData(BCData& bc, int i) const;
     void addBCData(BCData& bc);
     void addBCData(BCDataBase* bc);
     void setBoundaryIterator(std::vector<IntVector>& b, int i);
     void setInteriorIterator(std::vector<IntVector>& i, int ii);
     void setSFCXIterator(std::vector<IntVector>& i, int ii);
     void setSFCYIterator(std::vector<IntVector>& i, int ii);
     void setSFCZIterator(std::vector<IntVector>& i, int ii);
     void getBoundaryIterator(std::vector<IntVector>& b, int i) const;
     void getInteriorIterator(std::vector<IntVector>& i, int ii) const;
     void getSFCXIterator(std::vector<IntVector>& i, int ii) const;
     void getSFCYIterator(std::vector<IntVector>& i, int ii) const;
     void getSFCZIterator(std::vector<IntVector>& i, int ii) const;
     int getNumberChildren() const;
     BCDataBase* getChild(int i) const;
         
   private:
     std::vector<BCDataBase*> child;
   };

} // End namespace Uintah

#endif




