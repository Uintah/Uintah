#ifndef UINTAH_MPM_PRESSUREBC_H
#define UINTAH_MPM_PRESSUREBC_H

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using namespace SCIRun;

class GeometryPiece;
class ParticleCreator;
   
/**************************************

CLASS
   PressureBC
   
   Pressure Boundary Conditions for MPM
 
GENERAL INFORMATION

   PressureBC.h

   Biswajit Banerjee
   Department of Mechanical Engineering, University of Utah
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   Copyright (C) 2003 University of Utah

KEYWORDS
   PressureBC

DESCRIPTION
   Stores the pressure load curves and boundary imformation for
   pressure boundary conditions that can be applied to surfaces
   with simple geometry -  planes, cylinders and spheres.

WARNING
  
****************************************/

   class PressureBC : public MPMPhysicalBC  {

   public:

      // Construct a PressureBC object that contains
      // the area over which pressure is to be applied
      // and the value of that pressure (in the form
      // of a load curve)
      PressureBC(ProblemSpecP& ps);
      ~PressureBC();
      virtual std::string getType() const;

      // Locate and flag the material points to which this pressure BC is
      // to be applied. 
      bool flagMaterialPoint(const Point& p, const Vector& dxpp) const;
      
      // Get the load curve number for this pressure BC
      inline int loadCurveID() const {return d_loadCurve->getID();}

      // Get the surface 
      inline GeometryPiece* getSurface() const {return d_surface;}

      // Get the surface type
      inline std::string getSurfaceType() const {return d_surfaceType;}

      // Set the number of material points on the surface
      inline void numMaterialPoints(long num) {d_numMaterialPoints = num;}

      // Get the number of material points on the surface
      inline long numMaterialPoints() const {return d_numMaterialPoints;}

      // Get the area of the surface
      double getSurfaceArea() const;

      // Get the load curve 
      inline LoadCurve<double>* getLoadCurve() const {return d_loadCurve;}

      // Get the applied pressure at time t
      inline double pressure(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double forcePerParticle(double time) const;

      // Get the force vector to be applied at a point 
      Vector getForceVector(const Point& px, double forcePerParticle) const;

   private:

      // Prevent empty constructor
      PressureBC();

      // Prevent copying
      PressureBC(const PressureBC&);
      PressureBC& operator=(const PressureBC&);
      
      // Private Data
      // Surface information
      GeometryPiece* d_surface;
      std::string d_surfaceType;
      long d_numMaterialPoints;

      // Load curve information (Pressure and time)
      LoadCurve<double>* d_loadCurve;
   };
} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::PressureBC& bc);
#endif
