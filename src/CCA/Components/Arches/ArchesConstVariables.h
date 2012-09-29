/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- ArchesConstVariables.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesConstVariables_h
#define Uintah_Components_Arches_ArchesConstVariables_h

#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/StencilMatrix.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

/**************************************
CLASS
   ArchesConstVariables
   
   Class ArchesConstVariables creates and stores the VarConstVariables
   that are used in Arches

GENERAL INFORMATION
   ArchesConstVariables.h - declaration of the class
   
   Author: Stanislav Borodai (borodai@crsim.utah.edu), developed based on
   ArchesVariables
   
   Creation Date:   July 20, 2000

   C-SAFE 
   

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

namespace Uintah {

  class ArchesConstVariables {
    public:
      ArchesConstVariables();
      ~ArchesConstVariables();
      constCCVariable<double> density;
      constCCVariable<double> new_density;
      constCCVariable<double> density_guess;
      constCCVariable<double> denRefArray;
      constCCVariable<double> pressure;
      constCCVariable<double> viscosity;
      constCCVariable<double> scalar;
      constCCVariable<double> diffusivity;
      constCCVariable<double> drhodf;
      StencilMatrix<constCCVariable<double> > scalarDiffusionCoeff; //7 pt stl
      constCCVariable<double> scalarDiffNonlinearSrc;
      
      constSFCXVariable<double> uVelocity;
      constSFCYVariable<double> vVelocity;
      constSFCZVariable<double> wVelocity;
      
      constSFCXVariable<double> old_uVelocity;
      constSFCYVariable<double> old_vVelocity;
      constSFCZVariable<double> old_wVelocity;
      
      constCCVariable<double> old_density;
      constCCVariable<double> old_old_density;
      constCCVariable<int> cellType;
      constCCVariable<double> filterdrhodt; //add to the RHS of the pres eqn
      
      constSFCXVariable<double> uVelRhoHat;
      constSFCYVariable<double> vVelRhoHat;
      constSFCZVariable<double> wVelRhoHat;
      
      constCCVariable<double> divergence;
      constCCVariable<double> old_scalar;
      constCCVariable<double> enthalpy;
      constCCVariable<double> old_enthalpy;
      constCCVariable<double> temperature;
      constCCVariable<double> cp;
      constCCVariable<double> absorption;
      constCCVariable<double> sootFV;
      constCCVariable<double> h2o;
      constCCVariable<double> co2;
      constCCVariable<double> reactscalarSRC;
      StencilMatrix<constCCVariable<double> > scalarCoeff; //7 point stencil
      StencilMatrix<constCCVariable<double> > reactscalarCoeff;//7 pt stencil
      constCCVariable<double> scalarNonlinearSrc;
      constCCVariable<double> reactscalarNonlinearSrc;
      // New scalar Coef---
      constCCVariable<Stencil7> scalarTotCoef; 
      constCCVariable<Stencil7>  pressCoeff; //7 point stencil
      constCCVariable<double> pressNonlinearSrc;
      constCCVariable<double> voidFraction;
      
      constCCVariable<double> mmEnthSu;
      constCCVariable<double> mmEnthSp;
      
      constSFCXVariable<double> mmuVelSu;
      constSFCXVariable<double> mmuVelSp;
      constSFCYVariable<double> mmvVelSu;
      constSFCYVariable<double> mmvVelSp;
      constSFCZVariable<double> mmwVelSu;
      constSFCZVariable<double> mmwVelSp;

      constSFCXVariable<double> uFmms;
      constSFCXVariable<double> vFmms;
      constSFCXVariable<double> wFmms;

      constCCVariable<double> scalarBoundarySrc; 
      constCCVariable<double> enthalpyBoundarySrc;
      constSFCXVariable<double> umomBoundarySrc;
      constSFCYVariable<double> vmomBoundarySrc;
      constSFCZVariable<double> wmomBoundarySrc;

    }; // End class ArchesConstVariables
} // End namespace Uintah

#endif
