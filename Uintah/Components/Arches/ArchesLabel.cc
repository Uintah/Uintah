//----- ArchesLabel.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <SCICore/Util/NotFinished.h>

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Default constructor for ArchesLabel
//****************************************************************************
ArchesLabel::ArchesLabel()
{
  // The direction strings
  std::string dir[3];
  dir[0] = "XMom";
  dir[1] = "YMom";
  dir[2] = "ZMom";

  // Cell Information
  cellInfoLabel = scinew VarLabel("cellInformation",
			    PerPatch<CellInformation*>::getTypeDescription());
  // Cell type
  cellTypeLabel = scinew VarLabel("cellType", 
				  CCVariable<int>::getTypeDescription() );
  // Density Labels
  densityINLabel = scinew VarLabel("densityIN", 
				   CCVariable<double>::getTypeDescription() );
  densitySPLabel = scinew VarLabel("densitySP", 
				   CCVariable<double>::getTypeDescription() );
  densityCPLabel = scinew VarLabel("densityCP", 
				  CCVariable<double>::getTypeDescription() );
  // Viscosity Labels
  viscosityINLabel = scinew VarLabel("viscosityIN", 
				   CCVariable<double>::getTypeDescription() );
  viscosityCTSLabel = scinew VarLabel("viscosityCTS", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Labels
  pressureINLabel = scinew VarLabel("pressureIN", 
				    CCVariable<double>::getTypeDescription() );
  pressureSPBCLabel = scinew VarLabel("pressureSPBC", 
				      CCVariable<double>::getTypeDescription() );
  pressureCPBCLabel = scinew VarLabel("pressureCPBC", 
				      CCVariable<double>::getTypeDescription() );
  pressurePSLabel = scinew VarLabel("pressurePS", 
				    CCVariable<double>::getTypeDescription() );
  // Pressure Coeff Labels
  presCoefP0Label = scinew VarLabel("presCoefP0", 
				      CCVariable<double>::getTypeDescription() );
  presCoefP1Label = scinew VarLabel("presCoefP1", 
				      CCVariable<double>::getTypeDescription() );
  presCoefPBLMLabel = scinew VarLabel("presCoefPBLM", 
				      CCVariable<double>::getTypeDescription() );
  // Pressure Linear Src Labels
  presLinSrcP0Label = scinew VarLabel("presLinSrcP0", 
					CCVariable<double>::getTypeDescription() );
  presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM", 
					CCVariable<double>::getTypeDescription() );
  // Pressure Non Linear Src Labels
  presNonLinSrcP0Label = scinew VarLabel("presNonLinSrcP0", 
					   CCVariable<double>::getTypeDescription() );
  presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM", 
					   CCVariable<double>::getTypeDescription() );
  // U-Velocity Labels
  uVelocityINLabel = scinew VarLabel("uVelocityIN", 
				    SFCXVariable<double>::getTypeDescription() );
  uVelocitySPLabel = scinew VarLabel("uVelocitySP", 
				    SFCXVariable<double>::getTypeDescription() );
  uVelocitySPBCLabel = scinew VarLabel("uVelocitySPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC", 
				       SFCXVariable<double>::getTypeDescription() );
  uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  uVelocityMSLabel = scinew VarLabel("uVelocityMS", 
				       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Coeff Labels
  uVelCoefP0Label = scinew VarLabel("uVelCoefP0",
			       SFCXVariable<double>::getTypeDescription() );
  uVelCoefP1Label = scinew VarLabel("uVelCoefP1",
			       SFCXVariable<double>::getTypeDescription() );
  uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
			       SFCXVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    uVelCoefM0Label[ii] = scinew VarLabel("uVelCoefM0"+dir[ii],
			       SFCXVariable<double>::getTypeDescription() );
    uVelCoefM1Label[ii] = scinew VarLabel("uVelCoefM1"+dir[ii],
			       SFCXVariable<double>::getTypeDescription() );
    uVelCoefMBLMLabel[ii] = scinew VarLabel("uVelCoefMBLM"+dir[ii],
			       SFCXVariable<double>::getTypeDescription() );
  }
  // U-Velocity Convection Coeff Labels
  uVelConvCoefPBLMLabel = scinew VarLabel("uVelConvCoefPBLM",
				        SFCXVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    uVelConvCoefMBLMLabel[ii] = scinew VarLabel("uVelConvCoefMBLM"+dir[ii],
				        SFCXVariable<double>::getTypeDescription() );
  }
  // U-Velocity Linear Src Labels
  uVelLinSrcP0Label = scinew VarLabel("uVelLinSrcP0",
				 SFCXVariable<double>::getTypeDescription() );
  uVelLinSrcP1Label = scinew VarLabel("uVelLinSrcP1",
				 SFCXVariable<double>::getTypeDescription() );
  uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				 SFCXVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    uVelLinSrcM0Label[ii] = scinew VarLabel("uVelLinSrcM0"+dir[ii],
				 SFCXVariable<double>::getTypeDescription() );
    uVelLinSrcM1Label[ii] = scinew VarLabel("uVelLinSrcM1"+dir[ii],
				 SFCXVariable<double>::getTypeDescription() );
    uVelLinSrcMBLMLabel[ii] = scinew VarLabel("uVelLinSrcMBLM"+dir[ii],
				 SFCXVariable<double>::getTypeDescription() );
  }
  // U-Velocity Non Linear Src Labels
  uVelNonLinSrcP0Label = scinew VarLabel("uVelNonLinSrcP0",
				    SFCXVariable<double>::getTypeDescription() );
  uVelNonLinSrcP1Label = scinew VarLabel("uVelNonLinSrcP1",
				    SFCXVariable<double>::getTypeDescription() );
  uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    uVelNonLinSrcM0Label[ii] = scinew VarLabel("uVelNonLinSrcM0"+dir[ii],
				    SFCXVariable<double>::getTypeDescription() );
    uVelNonLinSrcM1Label[ii] = scinew VarLabel("uVelNonLinSrcM1"+dir[ii],
				    SFCXVariable<double>::getTypeDescription() );
    uVelNonLinSrcM2Label[ii] = scinew VarLabel("uVelNonLinSrcM2"+dir[ii],
				    SFCXVariable<double>::getTypeDescription() );
    uVelNonLinSrcMBLMLabel[ii] = scinew VarLabel("uVelNonLinSrcMBLM"+dir[ii],
				    SFCXVariable<double>::getTypeDescription() );
  }
  // V-Velocity Labels
  vVelocityINLabel = scinew VarLabel("vVelocityIN", 
				    SFCYVariable<double>::getTypeDescription() );
  vVelocitySPLabel = scinew VarLabel("vVelocitySP", 
				    SFCYVariable<double>::getTypeDescription() );
  vVelocitySPBCLabel = scinew VarLabel("vVelocitySPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC", 
				       SFCYVariable<double>::getTypeDescription() );
  vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  vVelocityMSLabel = scinew VarLabel("vVelocityMS", 
				       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Coeff Labels
  vVelCoefP0Label = scinew VarLabel("vVelCoefP0",
			       SFCYVariable<double>::getTypeDescription() );
  vVelCoefP1Label = scinew VarLabel("vVelCoefP1",
			       SFCYVariable<double>::getTypeDescription() );
  vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
			       SFCYVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    vVelCoefM0Label[ii] = scinew VarLabel("vVelCoefM0"+dir[ii],
			       SFCYVariable<double>::getTypeDescription() );
    vVelCoefM1Label[ii] = scinew VarLabel("vVelCoefM1"+dir[ii],
			       SFCYVariable<double>::getTypeDescription() );
    vVelCoefMBLMLabel[ii] = scinew VarLabel("vVelCoefMBLM"+dir[ii],
			       SFCYVariable<double>::getTypeDescription() );
  }
  // V-Velocity Convection Coeff Labels
  vVelConvCoefPBLMLabel = scinew VarLabel("vVelConvCoefPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    vVelConvCoefMBLMLabel[ii] = scinew VarLabel("vVelConvCoefMBLM"+dir[ii],
				   SFCYVariable<double>::getTypeDescription() );
  }
  // V-Velocity Linear Src Labels
  vVelLinSrcP0Label = scinew VarLabel("vVelLinSrcP0",
				 SFCYVariable<double>::getTypeDescription() );
  vVelLinSrcP1Label = scinew VarLabel("vVelLinSrcP1",
				 SFCYVariable<double>::getTypeDescription() );
  vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				 SFCYVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    vVelLinSrcM0Label[ii] = scinew VarLabel("vVelLinSrcM0"+dir[ii],
				 SFCYVariable<double>::getTypeDescription() );
    vVelLinSrcM1Label[ii] = scinew VarLabel("vVelLinSrcM1"+dir[ii],
				 SFCYVariable<double>::getTypeDescription() );
    vVelLinSrcMBLMLabel[ii] = scinew VarLabel("vVelLinSrcMBLM"+dir[ii],
				 SFCYVariable<double>::getTypeDescription() );
  }
  // V-Velocity Non Linear Src Labels
  vVelNonLinSrcP0Label = scinew VarLabel("vVelNonLinSrcP0",
				    SFCYVariable<double>::getTypeDescription() );
  vVelNonLinSrcP1Label = scinew VarLabel("vVelNonLinSrcP1",
				    SFCYVariable<double>::getTypeDescription() );
  vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    vVelNonLinSrcM0Label[ii] = scinew VarLabel("vVelNonLinSrcM0"+dir[ii],
				    SFCYVariable<double>::getTypeDescription() );
    vVelNonLinSrcM1Label[ii] = scinew VarLabel("vVelNonLinSrcM1"+dir[ii],
				    SFCYVariable<double>::getTypeDescription() );
    vVelNonLinSrcM2Label[ii] = scinew VarLabel("vVelNonLinSrcM2"+dir[ii],
				    SFCYVariable<double>::getTypeDescription() );
    vVelNonLinSrcMBLMLabel[ii] = scinew VarLabel("vVelNonLinSrcMBLM"+dir[ii],
				    SFCYVariable<double>::getTypeDescription() );
  }
  // W-Velocity Labels
  wVelocityINLabel = scinew VarLabel("wVelocityIN", 
				    SFCZVariable<double>::getTypeDescription() );
  wVelocitySPLabel = scinew VarLabel("wVelocitySP", 
				    SFCZVariable<double>::getTypeDescription() );
  wVelocitySPBCLabel = scinew VarLabel("wVelocitySPBC", 
				       SFCZVariable<double>::getTypeDescription() );
  wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC", 
				       SFCZVariable<double>::getTypeDescription() );
  wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC", 
				       SFCZVariable<double>::getTypeDescription() );
  wVelocityMSLabel = scinew VarLabel("wVelocityMS", 
				       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Coeff Labels
  wVelCoefP0Label = scinew VarLabel("wVelCoefP0",
			       SFCZVariable<double>::getTypeDescription() );
  wVelCoefP1Label = scinew VarLabel("wVelCoefP1",
			       SFCZVariable<double>::getTypeDescription() );
  wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
			       SFCZVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    wVelCoefM0Label[ii] = scinew VarLabel("wVelCoefM0"+dir[ii],
			       SFCZVariable<double>::getTypeDescription() );
    wVelCoefM1Label[ii] = scinew VarLabel("wVelCoefM1"+dir[ii],
			       SFCZVariable<double>::getTypeDescription() );
    wVelCoefMBLMLabel[ii] = scinew VarLabel("wVelCoefMBLM"+dir[ii],
			       SFCZVariable<double>::getTypeDescription() );
  }
  // W-Velocity Convection Coeff Labels
  wVelConvCoefPBLMLabel = scinew VarLabel("wVelConvCoefPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    wVelConvCoefMBLMLabel[ii] = scinew VarLabel("wVelConvCoefMBLM"+dir[ii],
				   SFCZVariable<double>::getTypeDescription() );
  }
  // W-Velocity Linear Src Labels
  wVelLinSrcP0Label = scinew VarLabel("wVelLinSrcP0",
				 SFCZVariable<double>::getTypeDescription() );
  wVelLinSrcP1Label = scinew VarLabel("wVelLinSrcP1",
				 SFCZVariable<double>::getTypeDescription() );
  wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				 SFCZVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    wVelLinSrcM0Label[ii] = scinew VarLabel("wVelLinSrcM0"+dir[ii],
				 SFCZVariable<double>::getTypeDescription() );
    wVelLinSrcM1Label[ii] = scinew VarLabel("wVelLinSrcM1"+dir[ii],
				 SFCZVariable<double>::getTypeDescription() );
    wVelLinSrcMBLMLabel[ii] = scinew VarLabel("wVelLinSrcMBLM"+dir[ii],
				 SFCZVariable<double>::getTypeDescription() );
  }
  // W-Velocity Non Linear Src Labels
  wVelNonLinSrcP0Label = scinew VarLabel("wVelNonLinSrcP0",
				    SFCZVariable<double>::getTypeDescription() );
  wVelNonLinSrcP1Label = scinew VarLabel("wVelNonLinSrcP1",
				    SFCZVariable<double>::getTypeDescription() );
  wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  for (int ii = 0; ii < 3; ++ii) {
    wVelNonLinSrcM0Label[ii] = scinew VarLabel("wVelNonLinSrcM0"+dir[ii],
				    SFCZVariable<double>::getTypeDescription() );
    wVelNonLinSrcM1Label[ii] = scinew VarLabel("wVelNonLinSrcM1"+dir[ii],
				    SFCZVariable<double>::getTypeDescription() );
    wVelNonLinSrcM2Label[ii] = scinew VarLabel("wVelNonLinSrcM2"+dir[ii],
				    SFCZVariable<double>::getTypeDescription() );
    wVelNonLinSrcMBLMLabel[ii] = scinew VarLabel("wVelNonLinSrcMBLM"+dir[ii],
				    SFCZVariable<double>::getTypeDescription() );
  }
  // Scalar 
  scalarINLabel = scinew VarLabel("scalarIN",
				    CCVariable<double>::getTypeDescription() );
  scalarSPLabel = scinew VarLabel("scalarSP",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Coef
  scalCoefSBLMLabel = scinew VarLabel("scalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Conv Coef
  scalConvCoefSBLMLabel = scinew VarLabel("scalConvCoefSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Linear Src
  scalLinSrcSBLMLabel = scinew VarLabel("scalLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Scalar Non Linear Src
  scalNonLinSrcSBLMLabel = scinew VarLabel("scalNonLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
  // Unsure stuff
  // Unsure stuff
  DUPBLMLabel = scinew VarLabel("DUPBLM",
				SFCXVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
ArchesLabel::~ArchesLabel()
{
}

//
// $Log$
// Revision 1.2  2000/07/19 06:30:01  bbanerje
// ** MAJOR CHANGES **
// If you want to get the old code go two checkins back.
//
// Revision 1.1  2000/07/18 22:33:48  bbanerje
// Changes to PressureSolver for put error. Added ArchesLabel.
//
//
