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



#ifndef Uintah_Components_Models_RadiationSolver_h
#define Uintah_Components_Models_RadiationSolver_h

#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Models/Radiation/RadiationVariables.h>
#include <CCA/Components/Models/Radiation/RadiationConstVariables.h>

#include <Core/Containers/Array1.h>

namespace Uintah {

class ProcessorGroup;
class LoadBalancer;
class RadiationVariables;
class RadiationConstVariables;
class ICELabel;
class Models_CellInformation;

using namespace SCIRun;

/**************************************
CLASS
   Models_RadiationSolver
   
   Class Models_RadiationSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   Models_RadiationSolver.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date: April 12, 2005
   
   C-SAFE 
   
   Copyright U of U 2005

KEYWORDS


DESCRIPTION
   Class Models_RadiationSolver is an abstract base class
   that solves the linearized PDE.  This routine is written based
   on RadiationSolver.h written by Gautham Krishnamoorthy (July 2, 2004) 
   (gautham@crsim.utah.edu) as part of Arches/Radiation.



WARNING
none
****************************************/

class Models_RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a LinearSolver.
      // PRECONDITIONS
      // POSTCONDITIONS
      Models_RadiationSolver();


      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~Models_RadiationSolver();


      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Setup the problem (etc.)
      virtual void problemSetup(const ProblemSpecP& params, bool shradiation) = 0;

      virtual void outputProblemSpec(ProblemSpecP& ps) = 0;


      ////////////////////////////////////////////////////////////////////////

   virtual void matrixCreate(const PatchSet* allpatches,
                             const PatchSubset* mypatches) = 0;

   virtual void setMatrix(const ProcessorGroup* pc,
                          const Patch* patch,
                          RadiationVariables* vars,
                          bool plusX, bool plusY, bool plusZ,
                          CCVariable<double>& SU,
                          CCVariable<double>& AB,
                          CCVariable<double>& AS,
                          CCVariable<double>& AW,
                          CCVariable<double>& AP,
                          CCVariable<double>& AE,
                          CCVariable<double>& AN,
                          CCVariable<double>& AT) = 0;

   virtual bool radLinearSolve() = 0;
   virtual void copyRadSoln(const Patch* patch, RadiationVariables* vars) = 0;
   virtual void destroyMatrix() = 0;

protected:

private:

}; // End class Models_RadiationSolver

} // End namespace Uintah

#endif  
