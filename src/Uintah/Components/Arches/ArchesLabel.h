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

      // Cell type
      const VarLabel* cellTypeLabel;

      // Density Labels
      const VarLabel* densityCPLabel;

      // Viscosity Labels
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
      const VarLabel* presLinSrcPBLMLabel;

      // Pressure Non Linear Src Labels
      const VarLabel* presNonLinSrcPBLMLabel;

      // U-Velocity Labels
      const VarLabel* uVelocitySPBCLabel;
      const VarLabel* uVelocitySIVBCLabel;

      // U-Velocity Coeff Labels
      const VarLabel* uVelCoefP0Label;
      const VarLabel* uVelCoefP1Label;
      const VarLabel* uVelCoefPBLMLabel;

      // U-Velocity Convection Coeff Labels
      const VarLabel* uVelConvCoefPBLMLabel;

      // U-Velocity Linear Src Labels
      const VarLabel* uVelLinSrcP0Label;
      const VarLabel* uVelLinSrcP1Label;
      const VarLabel* uVelLinSrcPBLMLabel;

      // U-Velocity Non Linear Src Labels
      const VarLabel* uVelNonLinSrcP0Label;
      const VarLabel* uVelNonLinSrcP1Label;
      const VarLabel* uVelNonLinSrcPBLMLabel;

      // V-Velocity Labels
      const VarLabel* vVelocitySPBCLabel;
      const VarLabel* vVelocitySIVBCLabel;

      // V-Velocity Coeff Labels
      const VarLabel* vVelCoefP0Label;
      const VarLabel* vVelCoefP1Label;
      const VarLabel* vVelCoefPBLMLabel;

      // V-Velocity Convection Coeff Labels
      const VarLabel* vVelConvCoefPBLMLabel;

      // V-Velocity Linear Src Labels
      const VarLabel* vVelLinSrcP0Label;
      const VarLabel* vVelLinSrcP1Label;
      const VarLabel* vVelLinSrcPBLMLabel;

      // V-Velocity Non Linear Src Labels
      const VarLabel* vVelNonLinSrcP0Label;
      const VarLabel* vVelNonLinSrcP1Label;
      const VarLabel* vVelNonLinSrcPBLMLabel;

      // W-Velocity Labels
      const VarLabel* wVelocitySPBCLabel;
      const VarLabel* wVelocitySIVBCLabel;

      // W-Velocity Coeff Labels
      const VarLabel* wVelCoefP0Label;
      const VarLabel* wVelCoefP1Label;
      const VarLabel* wVelCoefPBLMLabel;

      // W-Velocity Convection Coeff Labels
      const VarLabel* wVelConvCoefPBLMLabel;

      // W-Velocity Linear Src Labels
      const VarLabel* wVelLinSrcP0Label;
      const VarLabel* wVelLinSrcP1Label;
      const VarLabel* wVelLinSrcPBLMLabel;

      // W-Velocity Non Linear Src Labels
      const VarLabel* wVelNonLinSrcP0Label;
      const VarLabel* wVelNonLinSrcP1Label;
      const VarLabel* wVelNonLinSrcPBLMLabel;


    }; // End class ArchesLabel

  }  // End namespace ArchesSpace
}  // End namespace Uintah

#endif

//
// $Log$
// Revision 1.1  2000/07/18 22:33:50  bbanerje
// Changes to PressureSolver for put error. Added ArchesLabel.
//
//
