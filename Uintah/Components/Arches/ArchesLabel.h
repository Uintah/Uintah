//----- ArchesLabel.h -----------------------------------------------

#ifndef Uintah_Components_Arches_ArchesLabel_h
#define Uintah_Components_Arches_ArchesLabel_h

/**************************************
CLASS
   ArchesLabel
   
   Class ArchesLabel creates and stores the VarLabels that are used in Arches

GENERAL INFORMATION
   ArchesLabel.h - declaration of the class
   
   Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
   
   Creation Date:   July 18, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/

#include <Uintah/Grid/VarLabel.h>

namespace Uintah {
  namespace ArchesSpace {
    class ArchesLabel {
    public:

      ArchesLabel();
      ~ArchesLabel();

      // Cell Information
      const VarLabel* cellInfoLabel;

      // Cell type
      const VarLabel* cellTypeLabel;

      // Density Labels
      const VarLabel* densityINLabel;
      const VarLabel* densityCPLabel;
      const VarLabel* densitySPLabel;

      // Viscosity Labels
      const VarLabel* viscosityINLabel;
      const VarLabel* viscosityCTSLabel;

      // Pressure Labels
      const VarLabel* pressureINLabel;
      const VarLabel* pressureSPBCLabel;
      const VarLabel* pressureCPBCLabel;
      const VarLabel* pressurePSLabel;

      // Pressure Coeff Labels
      const VarLabel* presCoefP0Label;
      const VarLabel* presCoefP1Label;
      const VarLabel* presCoefPBLMLabel;

      // Pressure Linear Src Labels
      const VarLabel* presLinSrcP0Label;
      const VarLabel* presLinSrcPBLMLabel;

      // Pressure Non Linear Src Labels
      const VarLabel* presNonLinSrcP0Label;
      const VarLabel* presNonLinSrcPBLMLabel;

      // U-Velocity Labels
      const VarLabel* uVelocityINLabel;
      const VarLabel* uVelocitySPLabel;
      const VarLabel* uVelocitySPBCLabel;
      const VarLabel* uVelocitySIVBCLabel;
      const VarLabel* uVelocityCPBCLabel;
      const VarLabel* uVelocityMSLabel;

      // U-Velocity Coeff Labels
      const VarLabel* uVelCoefP0Label;
      const VarLabel* uVelCoefP1Label;
      const VarLabel* uVelCoefPBLMLabel;
      const VarLabel* uVelCoefM0Label[3];
      const VarLabel* uVelCoefM1Label[3];
      const VarLabel* uVelCoefMBLMLabel[3];

      // U-Velocity Convection Coeff Labels
      const VarLabel* uVelConvCoefPBLMLabel;
      const VarLabel* uVelConvCoefMBLMLabel[3];

      // U-Velocity Linear Src Labels
      const VarLabel* uVelLinSrcP0Label;
      const VarLabel* uVelLinSrcP1Label;
      const VarLabel* uVelLinSrcPBLMLabel;
      const VarLabel* uVelLinSrcM0Label[3];
      const VarLabel* uVelLinSrcM1Label[3];
      const VarLabel* uVelLinSrcMBLMLabel[3];

      // U-Velocity Non Linear Src Labels
      const VarLabel* uVelNonLinSrcP0Label;
      const VarLabel* uVelNonLinSrcP1Label;
      const VarLabel* uVelNonLinSrcPBLMLabel;
      const VarLabel* uVelNonLinSrcM0Label[3];
      const VarLabel* uVelNonLinSrcM1Label[3];
      const VarLabel* uVelNonLinSrcM2Label[3];
      const VarLabel* uVelNonLinSrcMBLMLabel[3];

      // V-Velocity Labels
      const VarLabel* vVelocityINLabel;
      const VarLabel* vVelocitySPLabel;
      const VarLabel* vVelocitySPBCLabel;
      const VarLabel* vVelocitySIVBCLabel;
      const VarLabel* vVelocityCPBCLabel;
      const VarLabel* vVelocityMSLabel;

      // V-Velocity Coeff Labels
      const VarLabel* vVelCoefP0Label;
      const VarLabel* vVelCoefP1Label;
      const VarLabel* vVelCoefPBLMLabel;
      const VarLabel* vVelCoefM0Label[3];
      const VarLabel* vVelCoefM1Label[3];
      const VarLabel* vVelCoefMBLMLabel[3];

      // V-Velocity Convection Coeff Labels
      const VarLabel* vVelConvCoefPBLMLabel;
      const VarLabel* vVelConvCoefMBLMLabel[3];

      // V-Velocity Linear Src Labels
      const VarLabel* vVelLinSrcP0Label;
      const VarLabel* vVelLinSrcP1Label;
      const VarLabel* vVelLinSrcPBLMLabel;
      const VarLabel* vVelLinSrcM0Label[3];
      const VarLabel* vVelLinSrcM1Label[3];
      const VarLabel* vVelLinSrcMBLMLabel[3];

      // V-Velocity Non Linear Src Labels
      const VarLabel* vVelNonLinSrcP0Label;
      const VarLabel* vVelNonLinSrcP1Label;
      const VarLabel* vVelNonLinSrcPBLMLabel;
      const VarLabel* vVelNonLinSrcM0Label[3];
      const VarLabel* vVelNonLinSrcM1Label[3];
      const VarLabel* vVelNonLinSrcM2Label[3];
      const VarLabel* vVelNonLinSrcMBLMLabel[3];

      // W-Velocity Labels
      const VarLabel* wVelocityINLabel;
      const VarLabel* wVelocitySPLabel;
      const VarLabel* wVelocitySPBCLabel;
      const VarLabel* wVelocitySIVBCLabel;
      const VarLabel* wVelocityCPBCLabel;
      const VarLabel* wVelocityMSLabel;

      // W-Velocity Coeff Labels
      const VarLabel* wVelCoefP0Label;
      const VarLabel* wVelCoefP1Label;
      const VarLabel* wVelCoefPBLMLabel;
      const VarLabel* wVelCoefM0Label[3];
      const VarLabel* wVelCoefM1Label[3];
      const VarLabel* wVelCoefMBLMLabel[3];

      // W-Velocity Convection Coeff Labels
      const VarLabel* wVelConvCoefPBLMLabel;
      const VarLabel* wVelConvCoefMBLMLabel[3];

      // W-Velocity Linear Src Labels
      const VarLabel* wVelLinSrcP0Label;
      const VarLabel* wVelLinSrcP1Label;
      const VarLabel* wVelLinSrcPBLMLabel;
      const VarLabel* wVelLinSrcM0Label[3];
      const VarLabel* wVelLinSrcM1Label[3];
      const VarLabel* wVelLinSrcMBLMLabel[3];

      // W-Velocity Non Linear Src Labels
      const VarLabel* wVelNonLinSrcP0Label;
      const VarLabel* wVelNonLinSrcP1Label;
      const VarLabel* wVelNonLinSrcPBLMLabel;
      const VarLabel* wVelNonLinSrcM0Label[3];
      const VarLabel* wVelNonLinSrcM1Label[3];
      const VarLabel* wVelNonLinSrcM2Label[3];
      const VarLabel* wVelNonLinSrcMBLMLabel[3];

      // Scalar Labels
      const VarLabel* scalarINLabel;
      const VarLabel* scalarSPLabel;

      // Scalar Coef
      const VarLabel* scalCoefSBLMLabel;

      // Scalar Conv Coef
      const VarLabel* scalConvCoefSBLMLabel;

      // Scalar Linear Src
      const VarLabel* scalLinSrcSBLMLabel;

      // Scalar NonLinear Src
      const VarLabel* scalNonLinSrcSBLMLabel;

      // Not sure what these labels are for
      const VarLabel* DUPBLMLabel;

    }; // End class ArchesLabel

  }  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.2  2000/07/19 06:30:01  bbanerje
// ** MAJOR CHANGES **
// If you want to get the old code go two checkins back.
//
// Revision 1.1  2000/07/18 22:33:50  bbanerje
// Changes to PressureSolver for put error. Added ArchesLabel.
//
//
