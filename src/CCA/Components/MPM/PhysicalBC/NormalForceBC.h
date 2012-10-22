/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_GRID_NormalForceBC_H
#define UINTAH_GRID_NormalForceBC_H

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   NormalForceBC
   
  
GENERAL INFORMATION

   NormalForceBC.h

   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
p   NormalForceBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NormalForceBC : public MPMPhysicalBC  {
   public:
      NormalForceBC(ProblemSpecP& ps);
      ~NormalForceBC();
      virtual std::string getType() const;
      virtual void outputProblemSpec(ProblemSpecP& ps);

      // Get the load curve
      inline LoadCurve<double>* getLoadCurve() const {return d_loadCurve;}

      // Get the applied pressure at time t
      inline double getLoad(double t) const {return d_loadCurve->getLoad(t);}

      // Get the force per particle at time t
      double forcePerParticle(double time) const;

   private:
      NormalForceBC(const NormalForceBC&);
      NormalForceBC& operator=(const NormalForceBC&);
      
      // Load curve information (Pressure and time)
      LoadCurve<double>* d_loadCurve;

   };
} // End namespace Uintah

#endif
