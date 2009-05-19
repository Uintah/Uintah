/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#ifndef Uintah_Components_Arches_RadLinearSolver_h
#define Uintah_Components_Arches_RadLinearSolver_h

#include <sci_defs/petsc_defs.h>

#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <Core/Containers/Array1.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
}
#endif

namespace Uintah {

class ProcessorGroup;

using namespace SCIRun;

/**************************************
CLASS
   RadLinearSolver
   
   Class RadLinearSolver uses gmres solver
   solver

GENERAL INFORMATION
   RadLinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class RadLinearSolver is a gmres linear solver

WARNING
   none

****************************************/

class RadLinearSolver: public RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a RadLinearSolver.
      RadLinearSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RadLinearSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      // to close petsc 
      void finalizeSolver();

      void matrixCreate(const PatchSet* allpatches,
                        const PatchSubset* mypatches);
                        
      void setMatrix(const ProcessorGroup* pc, 
                     const Patch* patch,
                     ArchesVariables* vars,
                     bool xplus, 
                     bool yplus, 
                     bool zplus,
                     CCVariable<double>& SU,
                     CCVariable<double>& AB,
                     CCVariable<double>& AS,
                     CCVariable<double>& AW,
                     CCVariable<double>& AP,
                     CCVariable<double>& AE,
                     CCVariable<double>& AN,
                     CCVariable<double>& AT);

      bool radLinearSolve();

      virtual void copyRadSoln(const Patch* patch, 
                               ArchesVariables* vars);
      virtual void destroyMatrix();
protected:

private:
      int numlrows;
      int numlcolumns;
      int globalrows;
      int globalcolumns;
      int d_nz;
      int o_nz;
      string d_pcType;
      string d_kspType;
      int d_overlap;
      int d_fill;
      int d_maxSweeps;
      bool d_shsolver;
      double d_tolerance; // convergence tolerence
      double d_initResid;
      double d_residual;
   const ProcessorGroup* d_myworld;
#ifdef HAVE_PETSC
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b, d_u;
#endif
}; // End class RadLinearSolver.h

} // End namespace Uintah

#endif  
  


