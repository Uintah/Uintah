#ifndef UINTAH_GRID_CrackBC_H
#define UINTAH_GRID_CrackBC_H

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   CrackBC
   
  
GENERAL INFORMATION

   CrackBC.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CrackBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CrackBC : public MPMPhysicalBC  {
   public:
      CrackBC(ProblemSpecP& ps);
      virtual std::string getType() const;

      double   x1() const;
      double   y1() const;
      double   x2() const;
      double   y2() const;
      double   x3() const;
      double   y3() const;
      double   x4() const;
      double   y4() const;

      const Vector&  e1() const;
      const Vector&  e2() const;
      const Vector&  e3() const;
      const Point&   origin() const;
         
   private:
      CrackBC(const CrackBC&);
      CrackBC& operator=(const CrackBC&);
      
      Point  d_origin;
      Vector d_e1,d_e2,d_e3;
      double d_x1,d_y1,d_x2,d_y2,d_x3,d_y3,d_x4,d_y4;
   };
   
} // end namespace Uintah

#endif
