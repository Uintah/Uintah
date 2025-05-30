/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_MPM_TORQUEBC_H
#define UINTAH_MPM_TORQUEBC_H

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
   TorqueBC
   
   Torque Boundary Conditions for MPM
 
GENERAL INFORMATION

   TorqueBC.h

   James Guilkey
   Department of Mechanical Engineering, University of Utah
   Laird Avenue Consulting, LLC

KEYWORDS
   TorqueBC

DESCRIPTION
   Stores the torque load curves and boundary information for
   torque boundary conditions that can be applied to surfaces
   with simple cylindrical geometry.

WARNING
  
****************************************/

   class TorqueBC : public MPMPhysicalBC  {

   public:

      // Construct a TorqueBC object that contains
      // the area over which torque is to be applied
      // and the value of that torque (in the form
      // of a load curve)
      TorqueBC(ProblemSpecP& ps, const GridP& grid, const MPMFlags* flags);
      ~TorqueBC();
      virtual std::string getType() const;

      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Locate and flag the material points to which this torque BC is
      // to be applied. 
      bool flagMaterialPoint(const Point& p, const Vector& dxpp);
      
      // Get the load curve number for this torque BC
      inline int loadCurveID() const {return d_loadCurve->getID();}

      // Get the load curve number for this torque BC
      inline int loadCurveMatl() const {return d_loadCurve->getMatl();}

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

      // Get the applied torque at time t
      inline double Torque(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double torquePerParticle(double time) const;

      // Get the force vector to be applied at a point 
      Vector getForceVector(const Point& px, double forcePerParticle,
                            const double time) const;

      // Get the force vector to be applied at 4 corners of the point 
      Vector getForceVectorCBDI(const Point& px, const Matrix3& psize,
                                const Matrix3& pDeformationMeasure,
                                double forcePerParticle, const double time,
                                Point& pExternalForceCorner1,
                                Point& pExternalForceCorner2,
                                Point& pExternalForceCorner3,
                                Point& pExternalForceCorner4,
                                const Vector& dxCell) const;

   private:

      // Prevent empty constructor
      TorqueBC();

      // Prevent copying
      TorqueBC(const TorqueBC&);
      TorqueBC& operator=(const TorqueBC&);
      
      // Private Data
      // Surface information
      GeometryPiece* d_surface;
      std::string d_surfaceType;
      long d_numMaterialPoints;
//      bool d_cylinder_end;
//      bool d_axisymmetric_end;
//      bool d_axisymmetric_side;
      bool d_outwardNormal;

      // Load curve information (Torque and time)
      LoadCurve<double>* d_loadCurve;

    public:
      Vector d_axis;
      //IntVector d_res;

      friend std::ostream& operator<<(std::ostream& out, const Uintah::TorqueBC& bc);
   };
} // End namespace Uintah

#endif
