#ifndef UINTAH_GRID_BCData_H
#define UINTAH_GRID_BCData_H

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

  class BCData  {
  public:
    BCData();
    ~BCData();
    BCData(const BCData&);
    BCData& operator=(const BCData&);
    bool operator==(const BCData&);
    bool operator<(const BCData&) const;
    void setBCValues(BoundCondBase* bc);
    const BoundCondBase* getBCValues(const string& type) const;
    void print() const;
    bool find(const string& type) const;
    
   private:
      // The map is for the name of the
    // bc type and then the actual bc data, i.e. 
    // "Velocity", VelocityBoundCond
    typedef map<string,BoundCondBase*>  bcDataType;
    bcDataType d_BCData;
    
  };
} // End namespace Uintah

#endif


