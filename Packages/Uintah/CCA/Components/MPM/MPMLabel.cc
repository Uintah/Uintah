#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

MPMLabel::MPMLabel()
{
  // Particle Variables

  //non PermanentParticleState
  pDeformationMeasureLabel = VarLabel::create("p.deformationMeasure",
			ParticleVariable<Matrix3>::getTypeDescription());

  pCrackEffectiveLabel = VarLabel::create("p.crackEffective",
			ParticleVariable<int>::getTypeDescription());

  pConnectivityLabel = VarLabel::create("p.connectivity",
			ParticleVariable<int>::getTypeDescription());

  pContactForceLabel = VarLabel::create("p.contactForce",
			ParticleVariable<Vector>::getTypeDescription());

  pXXLabel = VarLabel::create("p.positionXX",
			ParticleVariable<Point>::getTypeDescription());
  
  pStressLabel_afterFracture = VarLabel::create(
                        "p.stress_afterFracture",
			ParticleVariable<Matrix3>::getTypeDescription());

  pStressLabel_afterStrainRate = VarLabel::create(
                        "p.stress_afterStrainRate",
			ParticleVariable<Matrix3>::getTypeDescription());

  pVelocityLabel_afterFracture = VarLabel::create(
                        "p.velocity_afterFracture",
			ParticleVariable<Vector>::getTypeDescription());

  pVelocityLabel_afterUpdate = VarLabel::create(
                        "p.velocity_afterUpdate",
			ParticleVariable<Vector>::getTypeDescription());

  pStrainEnergyLabel = VarLabel::create("p.strainEnergy",
			ParticleVariable<double>::getTypeDescription());

  pRotationRateLabel = VarLabel::create("p.rotationRate",
			ParticleVariable<Vector>::getTypeDescription());

  pPressureLabel  = VarLabel::create( "p.pressure",
			ParticleVariable<double>::getTypeDescription() );
  
  //PermanentParticleState
  pStressLabel = VarLabel::create( "p.stress",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pCrackRadiusLabel = VarLabel::create( "p.CrackRadius",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel = VarLabel::create( "p.volume",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeDeformedLabel = VarLabel::create( "p.volumedeformed",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel = VarLabel::create( "p.mass",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel = VarLabel::create( "p.velocity", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel = VarLabel::create( "p.externalforce",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel = VarLabel::create("p.x",ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel = VarLabel::create( "p.temperature",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel = VarLabel::create( "p.temperatureGradient",
			ParticleVariable<Vector>::getTypeDescription() );

  pExternalHeatRateLabel = VarLabel::create( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = VarLabel::create( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel = VarLabel::create( "p.isBroken",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel = VarLabel::create( "p.crackNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel = VarLabel::create( "p.tipNormal",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel = VarLabel::create( "p.extensionDirection",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel = VarLabel::create( "p.toughness",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel = VarLabel::create( "p.energyReleaseRate",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel = VarLabel::create( "p.crackSurfacePressure",
			ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel = VarLabel::create( "p.displacement",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pIsIgnitedLabel  = VarLabel::create( "p.isIgnited",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel  = VarLabel::create( "p.massRate",
			ParticleVariable<double>::getTypeDescription() );
  
  // Particle Variables 
  pDeformationMeasureLabel_preReloc = VarLabel::create("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = VarLabel::create( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );

  pCrackRadiusLabel_preReloc = VarLabel::create( "p.CrackRadius+",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel_preReloc = VarLabel::create( "p.volume+",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel_preReloc = VarLabel::create( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel_preReloc = VarLabel::create( "p.velocity+", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel_preReloc = VarLabel::create( "p.externalforce+",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel_preReloc = VarLabel::create( "p.x+",
			ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);
  
  pTemperatureLabel_preReloc = VarLabel::create( "p.temperature+",
			ParticleVariable<double>::getTypeDescription() );
  
  pExternalHeatRateLabel_preReloc = VarLabel::create( "p.externalHeatRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = VarLabel::create( "p.surface+",
			ParticleVariable<int>::getTypeDescription() );

  pIsBrokenLabel_preReloc = VarLabel::create( "p.isBroken+",
			ParticleVariable<int>::getTypeDescription() );

  pCrackNormalLabel_preReloc = VarLabel::create( "p.crackNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pTipNormalLabel_preReloc = VarLabel::create( "p.tipNormal+",
			ParticleVariable<Vector>::getTypeDescription() );

  pExtensionDirectionLabel_preReloc = VarLabel::create( "p.extensionDirection+",
			ParticleVariable<Vector>::getTypeDescription() );

  pToughnessLabel_preReloc = VarLabel::create( "p.toughness+",
			ParticleVariable<double>::getTypeDescription() );

  pEnergyReleaseRateLabel_preReloc =VarLabel::create("p.energyReleaseRate+",
			ParticleVariable<double>::getTypeDescription() );

  pCrackSurfacePressureLabel_preReloc = 
                        VarLabel::create( "p.crackSurfacePressure+",
                        ParticleVariable<double>::getTypeDescription() );

  pDisplacementLabel_preReloc = VarLabel::create( "p.displacement+",
			ParticleVariable<Vector>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

  pIsIgnitedLabel_preReloc  = VarLabel::create( "p.isIgnited+",
			ParticleVariable<int>::getTypeDescription() );
  
  pMassRateLabel_preReloc  = VarLabel::create( "p.massRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  // Node Centered Variables
  
  gAccelerationLabel = VarLabel::create( "g.acceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedAccelerationLabel = VarLabel::create( "g.momexedacceleration",
			NCVariable<Vector>::getTypeDescription() );
  
  gMassLabel = VarLabel::create( "g.mass",
			NCVariable<double>::getTypeDescription() );
  
  gVelocityLabel = VarLabel::create( "g.velocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityLabel = VarLabel::create( "g.momexedvelocity",
			NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceLabel = VarLabel::create( "g.externalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceLabel = VarLabel::create( "g.internalforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarLabel = VarLabel::create( "g.velocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gMomExedVelocityStarLabel = VarLabel::create( "g.momexedvelocity_star",
			NCVariable<Vector>::getTypeDescription() );
  
  gTemperatureLabel = VarLabel::create("g.temperature",
			NCVariable<double>::getTypeDescription());

  gTemperatureNoBCLabel = VarLabel::create("g.temperaturenobc",
			NCVariable<double>::getTypeDescription());

  gTemperatureStarLabel = VarLabel::create("g.temperatureStar",
			NCVariable<double>::getTypeDescription());

  gTemperatureRateLabel = VarLabel::create("g.temperatureRate",
			NCVariable<double>::getTypeDescription());

  gInternalHeatRateLabel = VarLabel::create("g.internalHeatRate",
			NCVariable<double>::getTypeDescription());

  gExternalHeatRateLabel = VarLabel::create("g.externalHeatRate",
			NCVariable<double>::getTypeDescription());

  gThermalContactHeatExchangeRateLabel = 
     VarLabel::create("g.thermalContactHeatExchangeRate",
     NCVariable<double>::getTypeDescription());

  gNormTractionLabel = VarLabel::create( "g.normtraction",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = VarLabel::create( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

  gStressLabel   = VarLabel::create( "g.stress",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressForSavingLabel   = VarLabel::create( "g.stressFS",
                   NCVariable<Matrix3>::getTypeDescription() );

  gVolumeLabel     = VarLabel::create("g.volume",
			NCVariable<double>::getTypeDescription());

  gWeightLabel     = VarLabel::create("g.weight",
			NCVariable<double>::getTypeDescription());

  gradPAccNCLabel = VarLabel::create("gradPAccNC",
			NCVariable<Vector>::getTypeDescription());

  dTdt_NCLabel     = VarLabel::create("dTdt_NC",
			NCVariable<double>::getTypeDescription());

  massBurnFractionLabel  = VarLabel::create("massBurnFraction",
			NCVariable<double>::getTypeDescription());

  AccArchesNCLabel = VarLabel::create("AccArchesNC",
			NCVariable<Vector>::getTypeDescription() );

  frictionalWorkLabel = VarLabel::create("frictionalWork",
			NCVariable<double>::getTypeDescription());

  // Reduction variables
  partCountLabel = VarLabel::create("particleCount",
				   sumlong_vartype::getTypeDescription());

  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );

  StrainEnergyLabel = VarLabel::create( "StrainEnergy",
			sum_vartype::getTypeDescription() );

  KineticEnergyLabel = VarLabel::create( "KineticEnergy",
			sum_vartype::getTypeDescription() );

  TotalMassLabel = VarLabel::create( "TotalMass",
				 sum_vartype::getTypeDescription() );

  NTractionZMinusLabel = VarLabel::create( "NTractionZMinus",
			sum_vartype::getTypeDescription() );

  CenterOfMassPositionLabel = VarLabel::create( "CenterOfMassPosition",
				 sumvec_vartype::getTypeDescription() );

  CenterOfMassVelocityLabel = VarLabel::create( "CenterOfMassVelocity",
				 sumvec_vartype::getTypeDescription() );

  // for assigning particle ids
  pCellNAPIDLabel =
    VarLabel::create("cellNAPID", CCVariable<short int>::getTypeDescription());

  doMechLabel = VarLabel::create( "doMech", delt_vartype::getTypeDescription() );

} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  delete pDeformationMeasureLabel;
  delete pConnectivityLabel;
  delete pCrackEffectiveLabel;
  delete pContactForceLabel;
  delete pXXLabel;

  delete pStressLabel_afterFracture;
  delete pStressLabel_afterStrainRate;
  delete pVelocityLabel_afterFracture;
  delete pVelocityLabel_afterUpdate;

  delete pStrainEnergyLabel;
  delete pRotationRateLabel;

  //PermanentParticleState
  delete pStressLabel;
  delete pVolumeLabel;
  delete pVolumeDeformedLabel;
  delete pMassLabel;
  delete pVelocityLabel;
  delete pExternalForceLabel;
  delete pXLabel;
  delete pTemperatureLabel;
  delete pTemperatureGradientLabel;
  delete pExternalHeatRateLabel;
  delete pSurfLabel;
  delete pIsBrokenLabel;
  delete pCrackNormalLabel;
  delete pTipNormalLabel;
  delete pExtensionDirectionLabel;
  delete pToughnessLabel;
  delete pEnergyReleaseRateLabel;
  delete pCrackSurfacePressureLabel;
  delete pDisplacementLabel;
  delete pParticleIDLabel;
  delete pIsIgnitedLabel;
  delete pMassRateLabel;
  delete pPressureLabel;
  delete pCrackRadiusLabel;
  
  delete pDeformationMeasureLabel_preReloc;
  delete pStressLabel_preReloc;
  delete pVolumeLabel_preReloc;
  delete pMassLabel_preReloc;
  delete pVelocityLabel_preReloc;
  delete pExternalForceLabel_preReloc;
  delete pXLabel_preReloc;
  delete pTemperatureLabel_preReloc;
  delete pExternalHeatRateLabel_preReloc;
  delete pSurfLabel_preReloc;
  delete pIsBrokenLabel_preReloc;
  delete pCrackNormalLabel_preReloc;
  delete pTipNormalLabel_preReloc;
  delete pExtensionDirectionLabel_preReloc;
  delete pToughnessLabel_preReloc;
  delete pEnergyReleaseRateLabel_preReloc;
  delete pCrackSurfacePressureLabel_preReloc;
  delete pDisplacementLabel_preReloc;
  delete pParticleIDLabel_preReloc;
  delete pIsIgnitedLabel_preReloc;
  delete pMassRateLabel_preReloc;

  delete gAccelerationLabel;
  delete gMomExedAccelerationLabel;
  delete gMassLabel;
  delete gVelocityLabel;
  delete gMomExedVelocityLabel;
  delete gExternalForceLabel;
  delete gInternalForceLabel;
  delete gVelocityStarLabel;
  delete gMomExedVelocityStarLabel;
  delete gNormTractionLabel;
  delete gStressLabel;
  delete gSurfNormLabel;
  delete gTemperatureLabel;
  delete gTemperatureNoBCLabel;
  delete gTemperatureStarLabel;
  delete gTemperatureRateLabel;
  delete gInternalHeatRateLabel;
  delete gExternalHeatRateLabel;
  delete gThermalContactHeatExchangeRateLabel;
  delete gStressForSavingLabel;
  delete gVolumeLabel;
  delete gWeightLabel;
  delete gradPAccNCLabel;
  delete dTdt_NCLabel;
  delete massBurnFractionLabel;
  delete AccArchesNCLabel;
  delete frictionalWorkLabel;

  delete partCountLabel;
  delete delTLabel;
  delete doMechLabel;

  delete StrainEnergyLabel;
  delete KineticEnergyLabel;
  delete TotalMassLabel;
  delete NTractionZMinusLabel;
  delete CenterOfMassPositionLabel;
  delete CenterOfMassVelocityLabel;
  delete pCellNAPIDLabel;
}

void MPMLabel::registerPermanentParticleState(int i,
					      const VarLabel* label,
					      const VarLabel* preReloc_label)
{
  d_particleState[i].push_back(label);
  d_particleState_preReloc[i].push_back(preReloc_label);
  
}
