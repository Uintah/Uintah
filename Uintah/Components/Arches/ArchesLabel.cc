//----- ArchesLabel.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/ArchesLabel.h>
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
  // Cell type
  cellTypeLabel = scinew VarLabel("cellType", 
				  CCVariable<int>::getTypeDescription() );
  // Density Labels
  densityCPLabel = scinew VarLabel("densityCP", 
				  CCVariable<double>::getTypeDescription() );
  // Viscosity Labels
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
  presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM", 
					CCVariable<double>::getTypeDescription() );
  // Pressure Non Linear Src Labels
  presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM", 
					   CCVariable<double>::getTypeDescription() );
  // U-Velocity Labels
  uVelocitySPBCLabel = scinew VarLabel("uVelocitySPBC", 
				       SFCXVariable<double>::getTypeDescription() );
  uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC", 
				       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Coeff Labels
  uVelCoefP0Label = scinew VarLabel("uVelCoefP0",
			       SFCXVariable<double>::getTypeDescription() );
  uVelCoefP1Label = scinew VarLabel("uVelCoefP1",
			       SFCXVariable<double>::getTypeDescription() );
  uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
			       SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Convection Coeff Labels
  uVelConvCoefPBLMLabel = scinew VarLabel("uVelConvectCoefPBLM",
				        SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Linear Src Labels
  uVelLinSrcP0Label = scinew VarLabel("uVelLinSrcP0",
				 SFCXVariable<double>::getTypeDescription() );
  uVelLinSrcP1Label = scinew VarLabel("uVelLinSrcP1",
				 SFCXVariable<double>::getTypeDescription() );
  uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				 SFCXVariable<double>::getTypeDescription() );
  // U-Velocity Non Linear Src Labels
  uVelNonLinSrcP0Label = scinew VarLabel("uVelNonLinSrcP0",
				    SFCXVariable<double>::getTypeDescription() );
  uVelNonLinSrcP1Label = scinew VarLabel("uVelNonLinSrcP1",
				    SFCXVariable<double>::getTypeDescription() );
  uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  // V-Velocity Labels
  vVelocitySPBCLabel = scinew VarLabel("vVelocitySPBC", 
				       SFCYVariable<double>::getTypeDescription() );
  vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC", 
				       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Coeff Labels
  vVelCoefP0Label = scinew VarLabel("vVelCoefP0",
			       SFCYVariable<double>::getTypeDescription() );
  vVelCoefP1Label = scinew VarLabel("vVelCoefP1",
			       SFCYVariable<double>::getTypeDescription() );
  vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
			       SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Convection Coeff Labels
  vVelConvCoefPBLMLabel = scinew VarLabel("vVelConvectCoefPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Linear Src Labels
  vVelLinSrcP0Label = scinew VarLabel("vVelLinSrcP0",
				 SFCYVariable<double>::getTypeDescription() );
  vVelLinSrcP1Label = scinew VarLabel("vVelLinSrcP1",
				 SFCYVariable<double>::getTypeDescription() );
  vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				 SFCYVariable<double>::getTypeDescription() );
  // V-Velocity Non Linear Src Labels
  vVelNonLinSrcP0Label = scinew VarLabel("vVelNonLinSrcP0",
				    SFCYVariable<double>::getTypeDescription() );
  vVelNonLinSrcP1Label = scinew VarLabel("vVelNonLinSrcP1",
				    SFCYVariable<double>::getTypeDescription() );
  vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  // W-Velocity Labels
  wVelocitySPBCLabel = scinew VarLabel("wVelocitySPBC", 
				       SFCZVariable<double>::getTypeDescription() );
  wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC", 
				       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Coeff Labels
  wVelCoefP0Label = scinew VarLabel("wVelCoefP0",
			       SFCZVariable<double>::getTypeDescription() );
  wVelCoefP1Label = scinew VarLabel("wVelCoefP1",
			       SFCZVariable<double>::getTypeDescription() );
  wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
			       SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Convection Coeff Labels
  wVelConvCoefPBLMLabel = scinew VarLabel("wVelConvectCoefPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Linear Src Labels
  wVelLinSrcP0Label = scinew VarLabel("wVelLinSrcP0",
				 SFCZVariable<double>::getTypeDescription() );
  wVelLinSrcP1Label = scinew VarLabel("wVelLinSrcP1",
				 SFCZVariable<double>::getTypeDescription() );
  wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				 SFCZVariable<double>::getTypeDescription() );
  // W-Velocity Non Linear Src Labels
  wVelNonLinSrcP0Label = scinew VarLabel("wVelNonLinSrcP0",
				    SFCZVariable<double>::getTypeDescription() );
  wVelNonLinSrcP1Label = scinew VarLabel("wVelNonLinSrcP1",
				    SFCZVariable<double>::getTypeDescription() );
  wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
ArchesLabel::~ArchesLabel()
{
}

//
// $Log$
// Revision 1.1  2000/07/18 22:33:48  bbanerje
// Changes to PressureSolver for put error. Added ArchesLabel.
//
//
