#ifndef UINTAH_GRID_BoundCondData_H
#define UINTAH_GRID_BoundCondData_H

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>

namespace Uintah {
using std::map;
using std::string;


/**************************************

CLASS
   BoundCondData
   
   
GENERAL INFORMATION

   BoundCondData.h

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

  class BoundCondData {
  public:
    BoundCondData();
    ~BoundCondData();
    BoundCondData& operator=(const BoundCondData& rhs);
    BoundCondData(const BoundCondData& rhs);
    void setBCValues(int mat_id,BoundCondBase* bc);
    const BoundCondBase* getBCValues(int mat_id,const string& type) const;
    int getMatID() const;
    
   private:
    // The first map is for the material id (-1 for "all"), 
    // the second map is for the name of the
    // bc type and then the actual bc data, i.e. 
    // "Velocity", VelocityBoundCond
    typedef map<int,map<string,BoundCondBase* > > boundcondDataType;
    boundcondDataType d_data;
    
  };
} // End namespace Uintah

#endif


