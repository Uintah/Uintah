//----- ArchesVariables.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesVariables_h
#define Uintah_Components_Arches_ArchesVariables_h

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
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Components/Arches/StencilMatrix.h>


namespace Uintah {
  namespace ArchesSpace {
    class ArchesVariables {
    public:
      ArchesVariables();
      ~ArchesVariables();
      CCVariable<int> cellType;
      SFCXVariable<double> old_uVelocity;
      SFCYVariable<double> old_vVelocity;
      SFCZVariable<double> old_wVelocity;
      CCVariable<double> old_density;
      CCVariable<double> old_scalar;
      SFCXVariable<double> uVelocity;
      SFCYVariable<double> vVelocity;
      SFCZVariable<double> wVelocity;
      SFCXVariable<double> variableCalledDU; //used for pc calculation
      SFCYVariable<double> variableCalledDV; //used for pc calculation
      SFCZVariable<double> variableCalledDW; //used for pc calculation
      CCVariable<double> pressure;
      CCVariable<double> density;
      CCVariable<double> viscosity;
      CCVariable<double> scalar;
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
      StencilMatrix<CCVariable<double> > scalarCoeff; //7 point stencil
      StencilMatrix<CCVariable<double> > scalarConvectCoeff; //7 point stencil
      CCVariable<double> scalarLinearSrc;
      CCVariable<double> scalarNonlinearSrc;
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

    }; // End class ArchesVariables

  }  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.2.2.1  2000/10/26 10:05:13  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/09/29 20:32:36  rawat
// added underrelax to pressure solver
//
// Revision 1.2  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.1  2000/07/28 02:30:59  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
//
