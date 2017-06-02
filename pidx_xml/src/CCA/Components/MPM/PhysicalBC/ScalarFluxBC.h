/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_MPM_SCALARFLUX_H
#define UINTAH_MPM_SCALARFLUX_H

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Grid.h>
#include <Core/Math/Matrix3.h>
#include <iosfwd>

namespace Uintah {

class GeometryPiece;
class ParticleCreator;
   
/**************************************

CLASS
   ScalarFluxBC
   
   ScalarFlux Boundary Conditions for MPM
 
GENERAL INFORMATION

   ScalarFluxBC.h

   Jim Guilkey
   Department of Mechanical Engineering, University of Utah

KEYWORDS
   ScalarFluxBC

DESCRIPTION
   Stores the scalar flux load curves and boundary imformation for
   scalar flux boundary conditions that can be applied to surfaces
   with simple geometry -  planes, cylinders and spheres.

WARNING
  
****************************************/

   class ScalarFluxBC : public MPMPhysicalBC  {

   public:

      // Construct a ScalarFluxBC object that contains
      // the area over which scalar flux is to be applied
      // and the value of that scalar flux (in the form
      // of a load curve)
      ScalarFluxBC(ProblemSpecP& ps, const GridP& grid, const MPMFlags* flags);
      ~ScalarFluxBC();
      virtual std::string getType() const;

      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Locate and flag the material points to
      // which this scalar flux BC is to be applied. 
      bool flagMaterialPoint(const Point& p, const Vector& dxpp,
                                             Vector& areavec);
      
      // Get the load curve number for this scalar flux BC
      inline int loadCurveID() const {return d_loadCurve->getID();}

      // Get the surface 
      inline GeometryPiece* getSurface() const {return d_surface;}

      // Get the surface type
      inline std::string getSurfaceType() const {return d_surfaceType;}

      // Set the number of material points on the surface
      inline void numMaterialPoints(long num) {d_numMaterialPoints = num;}

      // Get the number of material points on the surface
      inline long numMaterialPoints() const {return d_numMaterialPoints;}

      // Get the load curve 
      inline LoadCurve<double>* getLoadCurve() const {return d_loadCurve;}

      // Get the applied scalar flux at time t
      inline double ScalarFlux(double t) const {return d_loadCurve->getLoad(t);}

#if 0
      // Get the area of the surface
      double getSurfaceArea() const;

      // Get the flux per particle at time t
      double fluxPerParticle(double time) const;
#endif

      // Get the flux for a particle at time t, given its area
      double fluxPerParticle(double time, double area) const;

   private:

      // Prevent empty constructor
      ScalarFluxBC();

      // Prevent copying
      ScalarFluxBC(const ScalarFluxBC&);
      ScalarFluxBC& operator=(const ScalarFluxBC&);
      
      // Private Data
      // Surface information
      GeometryPiece* d_surface;
      std::string d_surfaceType;
      long d_numMaterialPoints;
      bool d_cylinder_end;
      bool d_axisymmetric_end;
      bool d_axisymmetric_side;
      bool d_outwardNormal;

      // Load curve information (ScalarFlux and time)
      LoadCurve<double>* d_loadCurve;

    public:
      Vector d_dxpp;
      IntVector d_res;

      friend std::ostream& operator<<(std::ostream& out,
                                      const Uintah::ScalarFluxBC& bc);
   };
} // End namespace Uintah

#endif
