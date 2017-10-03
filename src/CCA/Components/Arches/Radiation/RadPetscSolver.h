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


#ifndef Uintah_Components_Arches_RadPetscSolver_h
#define Uintah_Components_Arches_RadPetscSolver_h

#include <sci_defs/petsc_defs.h>

#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>


#ifdef HAVE_PETSC

#if ((PETSC_VERSION_MAJOR == 3) && (PETSC_VERSION_MINOR < 4))
extern "C" {
#include "petscksp.h"
}
#else
#include "petscksp.h"
#endif

#endif

namespace Uintah {

class ProcessorGroup;

/**************************************
CLASS
   RadPetscSolver
   
   Class RadPetscSolver uses gmres solver
   solver

GENERAL INFORMATION
   RadPetscSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class RadPetscSolver is a gmres linear solver

WARNING
   none

****************************************/

class RadPetscSolver: public RadiationSolver {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a RadPetscSolver.
      RadPetscSolver(const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual Destructor
      virtual ~RadPetscSolver();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);


      void matrixCreate(const PatchSet* allpatches,
                        const PatchSubset* mypatches);

      void matrixInit(const Patch* patch){ };

      void gridSetup(bool plusX, bool plusY, bool plusZ){ };
                        
      void setMatrix(const ProcessorGroup* pc, 
                     const Patch* patch,
                     ArchesVariables* vars,
                     ArchesConstVariables* constvars,
                     bool xplus, 
                     bool yplus, 
                     bool zplus,
                     CCVariable<double>& SU,
                     CCVariable<double>& AB,
                     CCVariable<double>& AS,
                     CCVariable<double>& AW,
                     CCVariable<double>& AP,
                     const bool print );

      bool radLinearSolve( const int dir, const bool print_all_info );

      virtual void copyRadSoln(const Patch* patch, 
                               ArchesVariables* vars);
      virtual void destroyMatrix();
protected:

private:

      int d_numlrows;
      int d_numlcolumns;
      int d_globalrows;
      int d_globalcolumns;
      int d_nz;
      int o_nz;


      std::string d_pcType;
      std::string d_kspType;
      int d_overlap;
      int d_fill;
      int d_maxSweeps;
//      bool d_shsolver;
      double d_tolerance; // convergence tolerence
      const ProcessorGroup* d_myworld;
      
#ifdef HAVE_PETSC
   std::map<const Patch*, int> d_petscGlobalStart;
   std::map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b, d_u;
#endif
}; // End class RadPetscSolver.h

} // End namespace Uintah

#endif  
  


