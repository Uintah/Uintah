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



#ifndef Uintah_Components_Arches_LinearSolver_h
#define Uintah_Components_Arches_LinearSolver_h

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
namespace Uintah {

class ProcessorGroup;
class LoadBalancer;
class ArchesVariables;
class ArchesConstVariables;
class ArchesLabel;
class CellInformation;
// class StencilMatrix;

using namespace SCIRun;

/**************************************
CLASS
   LinearSolver
   
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.

GENERAL INFORMATION
   LinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class LinearSolver is an abstract base class
   that solves the linearized PDE.



WARNING
none
****************************************/

class LinearSolver {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of a LinearSolver.
  // PRECONDITIONS
  // POSTCONDITIONS
  LinearSolver();


  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual Destructor
  virtual ~LinearSolver();


  // GROUP: Problem Setup:
  ////////////////////////////////////////////////////////////////////////
  // Setup the problem (etc.)
  virtual void problemSetup(const ProblemSpecP& params) = 0;

  inline double getInitNorm() { return init_norm; }

  virtual void matrixCreate(const PatchSet* allpatches,
                            const PatchSubset* mypatches) = 0;
                            
  virtual void setMatrix(const ProcessorGroup* pc, 
                         const Patch* patch,
                         CCVariable<Stencil7>& coeff) = 0;

  virtual void setRHS_X(const ProcessorGroup* pc, 
                        const Patch* patch,
                        constCCVariable<double>& guess,
                        constCCVariable<double>& rhs,
                        bool construct_A ) = 0;


  virtual bool pressLinearSolve() = 0;
  virtual void copyPressSoln(const Patch* patch, ArchesVariables* vars) = 0;
  virtual void destroyMatrix() = 0;
  virtual void print(const string& desc, const int timestep, const int step) = 0;
  double init_norm;

protected:

private:

}; // End class LinearSolve

} // End namespace Uintah

#endif  
