#ifndef UINTAH_GRID_BoundCondReader_H
#define UINTAH_GRID_BoundCondReader_H

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/BCDataBase.h>
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
    void getBC(Patch::FaceType& face, BCData& bc);
    const BCDataArray getBC(Patch::FaceType& face) const;
    void combineBCS();
    
   private:
    map<Patch::FaceType,BCDataArray > d_bc;
  };
} // End namespace Uintah

#endif


