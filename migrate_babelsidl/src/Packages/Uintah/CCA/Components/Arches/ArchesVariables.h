//----- ArchesVariables.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesVariables_h
#define Uintah_Components_Arches_ArchesVariables_h

#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>

/**************************************
CLASS
   ArchesVariables
   
   Class ArchesVariables creates and stores the VarVariabless that are used in Arches

GENERAL INFORMATION
   ArchesVariables.h - declaration of the class
   
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

  class ArchesVariables {
    public:
      ArchesVariables();
      ~ArchesVariables();
      CCVariable<int> cellType;
      SFCXVariable<double> old_uVelocity;
      SFCYVariable<double> old_vVelocity;
      SFCZVariable<double> old_wVelocity;
      CCVariable<double> old_density;
      CCVariable<double> old_old_density;
      CCVariable<double> old_scalar;
      CCVariable<double> old_reactscalar;
      SFCXVariable<double> uVelocity;
      SFCYVariable<double> vVelocity;
      SFCZVariable<double> wVelocity;
      CCVariable<double> pressure;
      CCVariable<double> pressureNew;
      CCVariable<double> pressureCorr;
      CCVariable<double> density;
      CCVariable<double> pred_density;
      CCVariable<double> new_density; // using for pressure solver
      CCVariable<double> viscosity;
      CCVariable<double> scalar;
      CCVariable<double> reactscalar;
      CCVariable<double> enthalpy;
      CCVariable<double> old_enthalpy;
      StencilMatrix<SFCXVariable<double> > uVelocityCoeff;
      StencilMatrix<SFCYVariable<double> > vVelocityCoeff;
      StencilMatrix<SFCZVariable<double> > wVelocityCoeff;
      StencilMatrix<SFCXVariable<double> > uVelocityConvectCoeff;
      StencilMatrix<SFCYVariable<double> > vVelocityConvectCoeff;
      StencilMatrix<SFCZVariable<double> > wVelocityConvectCoeff;
      SFCXVariable<double> uVelLinearSrc; //SP term in Arches 
      SFCXVariable<double> uVelNonlinearSrc; // SU in Arches 
      SFCYVariable<double> vVelLinearSrc; //SP term in Arches 
      SFCYVariable<double> vVelNonlinearSrc; // SU in Arches 
      SFCZVariable<double> wVelLinearSrc; //SP term in Arches 
      SFCZVariable<double> wVelNonlinearSrc; // SU in Arches 
      StencilMatrix<CCVariable<double> > pressCoeff; //7 point stencil
      CCVariable<double> pressLinearSrc;
      CCVariable<double> pressNonlinearSrc;
      CCVariable<double> drhodt; //add to the RHS of the pressure eqn
      CCVariable<double> filterdrhodt; //add to the RHS of the pressure eqn
      StencilMatrix<CCVariable<double> > scalarCoeff; //7 point stencil
      StencilMatrix<CCVariable<double> > scalarDiffusionCoeff; //7 point stencil
      StencilMatrix<CCVariable<double> > scalarConvectCoeff; //7 point stencil
      CCVariable<double> scalarLinearSrc;
      CCVariable<double> scalarDiffNonlinearSrc;
      CCVariable<double> scalarNonlinearSrc;
      StencilMatrix<CCVariable<double> > reactscalarCoeff; //7 point stencil
      StencilMatrix<CCVariable<double> > reactscalarConvectCoeff; //7 point stencil
      CCVariable<double> reactscalarLinearSrc;
      CCVariable<double> reactscalarNonlinearSrc;
      // reaction source term
      CCVariable<double> reactscalarSRC;
      // for refdensity and refpressure
      double den_Ref;
      double press_ref;
      // added velocity hat vars
      SFCXVariable<double> uVelRhoHat;
      SFCYVariable<double> vVelRhoHat;
      SFCZVariable<double> wVelRhoHat;
      CCVariable<double> divergence;
      CCVariable<double> drhodf;
      CCVariable<double> densityPred;
      // for radiation calculations
      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> temperature;
      CCVariable<double> cp;
      CCVariable<double> absorption;
      CCVariable<double> sootFV;
      CCVariable<double> h2o;
      CCVariable<double> co2;
      CCVariable<double> ABSKG;
      CCVariable<double> ESRCG;
      CCVariable<double> src;
      CCVariable<double> cenint;
      CCVariable<double> shgamma;

      // multimaterial variables
      CCVariable<double> voidFraction;
      SFCXVariable<double> mmuVelSu;
      SFCXVariable<double> mmuVelSp;
      SFCYVariable<double> mmvVelSu;
      SFCYVariable<double> mmvVelSp;
      SFCZVariable<double> mmwVelSu;
      SFCZVariable<double> mmwVelSp;
      CCVariable<double> mmEnthSu;
      CCVariable<double> mmEnthSp;
      CCVariable<double> denRefArray;

    }; // End class ArchesVariables
   
} // End namespace Uintah

#endif
