//----- ArchesConstVariables.h -----------------------------------------------
#include <Packages/Uintah/Core/Exceptions/PetscError.h>

#ifndef Uintah_Components_Arches_ArchesConstVariables_h
#define Uintah_Components_Arches_ArchesConstVariables_h

#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>

/**************************************
CLASS
   ArchesConstVariables
   
   Class ArchesConstVariables creates and stores the VarConstVariabless that are used in Arches

GENERAL INFORMATION
   ArchesConstVariables.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   July 20, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

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
      constCCVariable<double> old_old_scalar;
      constCCVariable<double> enthalpy;
      constCCVariable<double> old_enthalpy;
      constCCVariable<double> old_old_enthalpy;
      constCCVariable<double> temperature;
      constCCVariable<double> cp;
      constCCVariable<double> absorption;
      constCCVariable<double> sootFV;
      constCCVariable<double> h2o;
      constCCVariable<double> co2;
      constCCVariable<double> c2h2;
      constCCVariable<double> reactscalarSRC;
      StencilMatrix<constCCVariable<double> > scalarCoeff; //7 point stencil
      StencilMatrix<constCCVariable<double> > reactscalarCoeff;//7 pt stencil
      constCCVariable<double> scalarNonlinearSrc;
      constCCVariable<double> reactscalarNonlinearSrc;
      StencilMatrix<constCCVariable<double> > pressCoeff; //7 point stencil
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

    }; // End class ArchesConstVariables
} // End namespace Uintah

#endif
