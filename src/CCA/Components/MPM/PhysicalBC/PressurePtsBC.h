/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef UINTAH_MPM_PRESSUREPTSBC_H
#define UINTAH_MPM_PRESSUREPTSBC_H

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
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
   PressurePtsBC
   
   Pressure Boundary Conditions for MPM
 
GENERAL INFORMATION

   PressurePtsBC.h

   Biswajit Banerjee
   Department of Mechanical Engineering, University of Utah
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

KEYWORDS
   PressurePtsBC

DESCRIPTION
   Stores the pressure load curves and boundary imformation for
   pressure boundary conditions that can be applied to surfaces
   where the geometry is described using points files

WARNING
  
****************************************/

   class PressurePtsBC : public MPMPhysicalBC  {

   public:

      // Construct a PressurePtsBC object that contains
      // the area over which pressure is to be applied
      // and the value of that pressure (in the form
      // of a load curve)
      PressurePtsBC(ProblemSpecP& ps, const GridP& grid, MPMFlags* flags);
      ~PressurePtsBC();
      virtual std::string getType() const;

      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Locate and flag the material points to which this pressure BC is
      // to be applied. 
      bool flagMaterialPoint(const Point& p, const Vector& dxpp);
      
      // Get the load curve number for this pressure BC
      inline int loadCurveID() const {return d_loadCurve->getID();}

      // Get the load curve number for this pressure BC
      inline int loadCurveMatl() const {return d_loadCurve->getMatl();}

      // Set the number of material points on the surface
      inline void numMaterialPoints(long num) {d_numMaterialPoints = num;}

      // Get the number of material points on the surface
      inline long numMaterialPoints() const {return d_numMaterialPoints;}

      // Get the area of the surface
      inline double getSurfaceArea() const {return d_surfaceArea;}

      // Get the load curve 
      inline LoadCurve<double>* getLoadCurve() const {return d_loadCurve;}

      // Get the applied pressure at time t
      inline double pressure(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double forcePerParticle(double time) const;

      // Get the force vector to be applied at a point 
      Vector getForceVector(const Point& px, double forcePerParticle,
                            const double time) const;

   private:

      // Prevent empty constructor
      PressurePtsBC();

      // Prevent copying
      PressurePtsBC(const PressurePtsBC&);
      PressurePtsBC& operator=(const PressurePtsBC&);
      
      // Private Data
      // Surface information
      long d_numMaterialPoints;
      double d_surfaceArea;

      // Load curve information (Pressure and time)
      LoadCurve<double>* d_loadCurve;

    public:
      Vector d_dxpp;
      IntVector d_res;

      friend std::ostream& operator<<(std::ostream& out, const Uintah::PressurePtsBC& bc);
   };
} // End namespace Uintah

#endif
