#ifndef UINTAH_MPM_HEATFLUXBC_H
#define UINTAH_MPM_HEATFLUXBC_H

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
   HeatFluxBC
   
   HeatFlux Boundary Conditions for MPM
 
GENERAL INFORMATION

   HeatFluxBC.h

   Biswajit Banerjee
   Department of Mechanical Engineering, University of Utah
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   Copyright (C) 2003 University of Utah

KEYWORDS
   HeatFluxBC

DESCRIPTION
   Stores the heatflux load curves and boundary imformation for
   heatflux boundary conditions that can be applied to surfaces
   with simple geometry -  planes, cylinders and spheres.

WARNING
  
****************************************/

   class HeatFluxBC : public MPMPhysicalBC  {

   public:

      // Construct a HeatFluxBC object that contains
      // the area over which heatflux is to be applied
      // and the value of that heatflux (in the form
      // of a load curve)
      HeatFluxBC(ProblemSpecP& ps);
      ~HeatFluxBC();
      virtual std::string getType() const;

      // Locate and flag the material points to which this heatflux BC is
      // to be applied. 
      bool flagMaterialPoint(const Point& p, const Vector& dxpp) const;
      
      // Get the load curve number for this heatflux BC
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

      // Get the applied heatflux at time t
      inline double heatflux(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double forcePerParticle(double time) const;

      // Get the force vector to be applied at a point 
      Vector getForceVector(const Point& px, double forcePerParticle) const;

   private:

      // Prevent empty constructor
      HeatFluxBC();

      // Prevent copying
      HeatFluxBC(const HeatFluxBC&);
      HeatFluxBC& operator=(const HeatFluxBC&);
      
      // Private Data
      // Surface information
      GeometryPiece* d_surface;
      std::string d_surfaceType;
      long d_numMaterialPoints;

      // Load curve information (HeatFlux and time)
      LoadCurve<double>* d_loadCurve;

      friend std::ostream& operator<<(std::ostream& out, const Uintah::HeatFluxBC& bc);
   };
} // End namespace Uintah

#endif
