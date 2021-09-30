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

#ifndef UINTAH_MPM_HydrostaticBC_H
#define UINTAH_MPM_HydrostaticBC_H

#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>

#include <Core/Grid/Variables/CCVariable.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Grid.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <iosfwd>

namespace Uintah {

class GeometryPiece;
class ParticleCreator;

/**************************************

CLASS
   HydrostaticBC

   Pressure Boundary Conditions for MPM

GENERAL INFORMATION

   HydrostaticBC.h

   Extended from the other BCs.

   Hilde Aas Nøst
   NTNU Norwegian University of Science and Technology

KEYWORDS
   HydrostaticBC

DESCRIPTION
   Stores the hydrostatic boundary condition that can be applied to surfaces
   with simple geometry -  planes, cylinders and spheres.

WARNING

****************************************/

class HydrostaticBC : public MPMPhysicalBC {
 public:
  // Construct a HydrostaticBC object that contains
  // the area over which pressure is to be applied
  // and the value of that pressure (in the form
  // of a load curve)
  HydrostaticBC(ProblemSpecP& ps, const GridP& grid, const MPMFlags* flags);
  ~HydrostaticBC();
  virtual std::string getType() const;

  virtual void outputProblemSpec(ProblemSpecP& ps);

  // Locate and flag the material points to which this pressure BC is
  // to be applied.
  bool flagMaterialPoint(const Point& p, const Vector& dxpp);

  double getCellAveragePorePressure(const IntVector cellindex,
                                    const Patch* patch) const;
  // Return the load curve ID of the hydrostatic loadcurve
  // Fixed value of 999 for now
  inline int loadCurveID() const { return 999; }

  // Get the surface
  inline GeometryPiece* getSurface() const { return d_surface; }

  // Get the surface type
  inline std::string getSurfaceType() const { return d_surfaceType; }

  // Set the number of material points on the surface
  inline void numPtsPerCell(constCCVariable<int>& num) {
    CCnumPtsPerCell.copyData(num);
  }

  // Get the number of material points on the surface
  inline constCCVariable<int> numPtsPerCell() const { return CCnumPtsPerCell; }

  // Set the number of material points on the surface
  inline void numMaterialPoints(long num) { d_numMaterialPoints = num; }

  // Get the number of material points on the surface
  inline long numMaterialPoints() const { return d_numMaterialPoints; }

  Vector getSurfaceNormal() const;

  // Get the cell surface area
  double getCellSurfaceArea(const Patch* patch) const;

  // Get the area of the surface
  double getSurfaceArea() const;

  // Get the hydrostatic pressure
  inline double getHydrostatic() const { return d_hydrostatic; }

 private:
  // Prevent empty constructor
  HydrostaticBC();

  // Prevent copying
  HydrostaticBC(const HydrostaticBC&);
  HydrostaticBC& operator=(const HydrostaticBC&);

  // Private Data
  // Surface information
  GeometryPiece* d_surface;
  std::string d_surfaceType;
  long d_numMaterialPoints;
  bool d_cylinder_end;
  bool d_axisymmetric_end;
  bool d_axisymmetric_side;
  bool d_outwardNormal;

  // Number of boundary particles per cell
  CCVariable<int> CCnumPtsPerCell;

  // Hydrostatic pressure
  double d_hydrostatic;

 public:
  Vector d_dxpp;
  IntVector d_res;

  friend std::ostream& operator<<(std::ostream& out,
                                  const Uintah::HydrostaticBC& bc);
};
}  // End namespace Uintah

#endif
