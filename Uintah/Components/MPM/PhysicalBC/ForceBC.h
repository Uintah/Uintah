#ifndef UINTAH_GRID_ForceBC_H
#define UINTAH_GRID_ForceBC_H

#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/BBox.h>
#include <Uintah/Interface/ProblemSpecP.h>

using namespace Uintah;

namespace Uintah {
namespace MPM {

  using SCICore::Geometry::Vector;
  using SCICore::Geometry::BBox;
   
/**************************************

CLASS
   ForceBC
   
  
GENERAL INFORMATION

   ForceBC.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ForceBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ForceBC : public MPMPhysicalBC  {
   public:
      ForceBC(ProblemSpecP& ps);
      virtual std::string getType() const;

      const Vector&  getForceDensity() const;
      const Vector&   getLowerRange() const;
      const Vector&   getUpperRange() const;
         
   private:
      ForceBC(const ForceBC&);
      ForceBC& operator=(const ForceBC&);
      
      Vector d_forceDensity;
      Vector d_lowerRange;
      Vector d_upperRange;
   };
   
} // end namespace MPM
} // end namespace Uintah

#endif

// $Log$
// Revision 1.1  2000/08/07 00:42:42  tan
// Added MPMPhysicalBC class to handle all kinds of physical boundary conditions
// in MPM.  Currently implemented force boundary conditions.
//
//
