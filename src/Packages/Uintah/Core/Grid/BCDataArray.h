#ifndef UINTAH_GRID_BCDataArray_H
#define UINTAH_GRID_BCDataArray_H

#include <Packages/Uintah/Core/Grid/BCData.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using std::vector;
  using std::map;

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
     ~BCDataArray();
     BCDataArray(const BCDataArray& bc);
     BCDataArray& operator=(const BCDataArray& bc);

     BCDataArray* clone();
     const BoundCondBase* getBoundCondData(int mat_id,string type, int i) const;
     void addBCData(int mat_id,BCGeomBase* bc);
     void setBoundaryIterator(int mat_id,vector<IntVector>& b, int i);
     void setNBoundaryIterator(int mat_id,vector<IntVector>& b, int i);
     void setInteriorIterator(int mat_id,vector<IntVector>& i, int ii);
     void getBoundaryIterator(int mat_id,vector<IntVector>& b, int i) const;
     void getNBoundaryIterator(int mat_id,vector<IntVector>& b, int i) const;
     void getInteriorIterator(int mat_id,vector<IntVector>& i, int ii) const;
     int getNumberChildren(int mat_id) const;
     BCGeomBase* getChild(int mat_id,int i) const;
     void print();

     typedef map<int,vector<BCGeomBase*> > bcDataArrayType;         
   private:
     // The map is for the mat_id.  -1 is for mat_id = "all", 0, for 
     // mat_id = "0", etc.
     bcDataArrayType d_BCDataArray;
     friend class Patch;
     friend class BCReader;
   };

} // End namespace Uintah

#endif




