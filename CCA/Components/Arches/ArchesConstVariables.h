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
// next 3 for filtering
    StencilMatrix<constSFCXVariable<double> > filteredRhoUjU; //SFCX 3 element vector
    StencilMatrix<constSFCYVariable<double> > filteredRhoUjV; //SFCY 3 element vector
    StencilMatrix<constSFCZVariable<double> > filteredRhoUjW; //SFCZ 3 element vector
// next 6 to make old stuff compile, actually not used
      StencilMatrix<constSFCXVariable<double> > uVelocityCoeff;
      StencilMatrix<constSFCYVariable<double> > vVelocityCoeff;
      StencilMatrix<constSFCZVariable<double> > wVelocityCoeff;
      constSFCXVariable<double> uVelNonlinearSrc; // SU in Arches 
      constSFCYVariable<double> vVelNonlinearSrc; // SU in Arches 
      constSFCZVariable<double> wVelNonlinearSrc; // SU in Arches 

/*


      CCVariable<double> old_reactscalar;
      SFCXVariable<double> variableCalledDU; //used for pc calculation
      SFCYVariable<double> variableCalledDV; //used for pc calculation
      SFCZVariable<double> variableCalledDW; //used for pc calculation
      CCVariable<double> pressureNew;
      CCVariable<double> pressureCorr;
      CCVariable<double> pred_density;
      CCVariable<double> reactscalar;
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
      CCVariable<double> pressLinearSrc;
      CCVariable<double> drhodt; //add to the RHS of the pressure eqn
      StencilMatrix<CCVariable<double> > scalarConvectCoeff; //7 point stencil
      CCVariable<double> scalarLinearSrc;
      StencilMatrix<CCVariable<double> > reactscalarCoeff; //7 point stencil
      StencilMatrix<CCVariable<double> > reactscalarConvectCoeff; //7 point stencil
      CCVariable<double> reactscalarLinearSrc;
      CCVariable<double> reactscalarNonlinearSrc;
      // reaction source term
      // for refdensity and refpressure
      double den_Ref;
      double press_ref;
      // for residual calculations
      double residPress;
      double truncPress;
      double residUVel;
      double truncUVel;
      double residVVel;
      double truncVVel;
      double residWVel;
      double truncWVel;
      double residScalar;
      double truncScalar;
      CCVariable<double> residualPressure;
      SFCXVariable<double> residualUVelocity;
      SFCYVariable<double> residualVVelocity;
      SFCZVariable<double> residualWVelocity;
      CCVariable<double> residualScalar;      
      CCVariable<double> residualReactivescalar;      
      CCVariable<double> residualEnthalpy;      
      // pressure gradient vars added to momentum source terms
      SFCXVariable<double> pressGradUSu;
      SFCYVariable<double> pressGradVSu;
      SFCZVariable<double> pressGradWSu;
      // added velocity hat vars
      CCVariable<double> densityPred;
      // for radiation calculations
      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> ABSKG;
      CCVariable<double> ESRCG;
      CCVariable<double> src;
      CCVariable<double> cenint;

      // multimaterial variables
// variables for filtered convection fluxes
*/
    }; // End class ArchesConstVariables
} // End namespace Uintah

#endif
