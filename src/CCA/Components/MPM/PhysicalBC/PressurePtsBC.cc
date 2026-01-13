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

#include <CCA/Components/MPM/PhysicalBC/PressurePtsBC.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/Matrix3.h>
#include <iostream>

using namespace Uintah;
using namespace std;

// Store the geometry object and the load curve
PressurePtsBC::PressurePtsBC(ProblemSpecP& ps, const GridP& grid,
                             MPMFlags* flags)
{
  // User must specify initial surface area
  d_surfaceArea = 0;  // this value is read in on a restart
  ps->require("SurfaceArea",d_surfaceArea);

  d_numMaterialPoints = 0;  // this value is read in on a restart
  ps->get("numberOfParticlesOnLoadSurface",d_numMaterialPoints);

  // Read and save the load curve information
  d_loadCurve = scinew LoadCurve<double>(ps);

  flags->d_useParticleNormals = true;

  //__________________________________
  //   Bulletproofing
}

// Destroy the pressure BCs
PressurePtsBC::~PressurePtsBC()
{
  delete d_loadCurve;
}

void PressurePtsBC::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP press_ps = ps->appendChild("pressure_pts");
  press_ps->appendElement("SurfaceArea",                   d_surfaceArea);
  press_ps->appendElement("numberOfParticlesOnLoadSurface",d_numMaterialPoints);
}

// Get the type of this object for BC application
std::string 
PressurePtsBC::getType() const
{
  return "PressurePts";
}

// Locate and flag the material points to which this pressure BC is
// to be applied. Assumes that the "checkForSurface" function in ParticleCreator.cc
// has been used to identify this material point as being on the surface of the body.
// WARNING : For this logic to work, the surface object should be a 
// box (zero volume), cylinder, sphere geometry piece that touches
// contains the surface on which the pressure is to be applied.
bool
PressurePtsBC::flagMaterialPoint(const Point& p, 
                                 const Vector& dxpp)
{
  bool flag = false;

  return flag;
}

// Calculate the force per particle at a certain time
double 
PressurePtsBC::forcePerParticle(double time) const
{
  if (d_numMaterialPoints < 1) return 0.0;

  // Get the initial pressure that is applied ( t = 0.0 )
  double press = pressure(time);

  // Calculate the forec per particle
  return (press*d_surfaceArea)/static_cast<double>(d_numMaterialPoints);
}

// Calculate the force vector to be applied to a particular
// material point location
Vector
PressurePtsBC::getForceVector(const Point& pnormalAsPoint, 
                                    double forcePerParticle,
                              const double time) const
{
  // This shouldn't get called
  Vector force(0.0,0.0,0.0);

  return force;
}

namespace Uintah {
// A method to print out the pressure bcs
ostream& operator<<(ostream& out, const PressurePtsBC& bc) 
{
   out << "Begin MPM Pressure Pts BC # = " << bc.loadCurveID() << endl;
   out << "    Surface Area = " << bc.getSurfaceArea() << endl;
   out << "    Time vs. Load = " << endl;
   LoadCurve<double>* lc = bc.getLoadCurve();
   int numPts = lc->numberOfPointsOnLoadCurve();
   for (int ii = 0; ii < numPts; ++ii) {
     out << "        time = " << lc->getTime(ii) 
         << " pressure = " << lc->getLoad(ii) << endl;
   }
   out << "End MPM Pressure Pts BC # = " << bc.loadCurveID() << endl;
   return out;
}

} // end namespace Uintah
