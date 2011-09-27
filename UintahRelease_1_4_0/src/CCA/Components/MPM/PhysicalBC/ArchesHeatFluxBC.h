/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_MPM_ARCHES_HEATFLUXBC_H
#define UINTAH_MPM_ARCHES_HEATFLUXBC_H

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/MPM/PhysicalBC/PolynomialData.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Grid.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <iosfwd>

namespace Uintah {

using namespace SCIRun;

class GeometryPiece;
class ParticleCreator;
   
/**************************************

CLASS
   ArchesHeatFluxBC
   
   HeatFlux Boundary Conditions for MPM
 
GENERAL INFORMATION

   ArchesHeatFluxBC.h

   Biswajit Banerjee
   Department of Mechanical Engineering, University of Utah
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   Copyright (C) 2003 University of Utah

KEYWORDS
   ArchesHeatFluxBC

DESCRIPTION
   Stores the heatflux load curves and boundary imformation for
   heatflux boundary conditions that can be applied to surfaces
   with simple geometry -  planes, cylinders and spheres.

WARNING
  
****************************************/

   class ArchesHeatFluxBC : public MPMPhysicalBC  {

   public:

      // Construct a ArchesHeatFluxBC object that contains
      // the area over which heatflux is to be applied
      // and the value of that heatflux (in the form
      // of a load curve)
      ArchesHeatFluxBC(ProblemSpecP& ps,const GridP& grid);
      ~ArchesHeatFluxBC();
      virtual std::string getType() const;
      virtual void outputProblemSpec(ProblemSpecP& ps);

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

      // Get the flux per particle at time t
      double fluxPerParticle(double time) const;

      // Get the flux vector to be applied at a point 
      virtual double getFlux(const Point& px, double forcePerParticle) const;

   private:

      // Prevent empty constructor
      ArchesHeatFluxBC();

      // Prevent copying
      ArchesHeatFluxBC(const ArchesHeatFluxBC&);
      ArchesHeatFluxBC& operator=(const ArchesHeatFluxBC&);
      
      // Private Data
      // Surface information
      GeometryPiece* d_surface;
      std::string d_surfaceType;
      long d_numMaterialPoints;

      // Load curve information (HeatFlux and time)
      LoadCurve<double>* d_loadCurve;

      PolynomialData* d_polyData;

      friend std::ostream& operator<<(std::ostream& out, const Uintah::ArchesHeatFluxBC& bc);
   };
} // End namespace Uintah

#endif
