#ifndef UINTAH_GRID_BoundCondReader_H
#define UINTAH_GRID_BoundCondReader_H

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCData.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

namespace Uintah {
  using std::map;
  using std::string;
  using std::vector;

/**************************************

CLASS
   BoundCondReader
   
   
GENERAL INFORMATION

   BoundCondReader.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2001 SCI Group

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BCReader  {
  public:
    BCReader();
    ~BCReader();
    void read(ProblemSpecP& ps);
    BCGeomBase* createBoundaryConditionFace(ProblemSpecP& ps,
					    Patch::FaceType& face_side);
    void getBC(Patch::FaceType& face, BoundCondData& bc);
    const BCDataArray getBCDataArray(Patch::FaceType& face) const;
    void combineBCS();
    bool compareBCData(BCGeomBase* b1, BCGeomBase* b2);

   private:
    friend class Level;
    friend class Patch;
    map<Patch::FaceType,BCDataArray > d_BCReaderData;
    vector<BoundCondData> d_bcs;
  };

  void print(BCGeomBase* p);

} // End namespace Uintah

#endif


