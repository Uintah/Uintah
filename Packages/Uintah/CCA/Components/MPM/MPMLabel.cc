#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using namespace Uintah;
using std::cerr;
using std::endl;

MPMLabel::MPMLabel()
{
  // Particle Variables

  //non PermanentParticleState
  pPressureLabel  = VarLabel::create( "p.pressure",
			ParticleVariable<double>::getTypeDescription() );
  
  pTemperatureGradientLabel = VarLabel::create( "p.temperatureGradient",
			ParticleVariable<Vector>::getTypeDescription() );

  p_qLabel  = VarLabel::create( "p.q",
			ParticleVariable<double>::getTypeDescription() );
  
  //PermanentParticleState
  pDeformationMeasureLabel = VarLabel::create("p.deformationMeasure",
			ParticleVariable<Matrix3>::getTypeDescription());

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
  
  pXLabel = VarLabel::create("p.x",
			     ParticleVariable<Point>::getTypeDescription(),
			     IntVector(0,0,0), VarLabel::PositionVariable);
  
  pTemperatureLabel = VarLabel::create( "p.temperature",
			ParticleVariable<double>::getTypeDescription() );
                     
  pSp_volLabel      = VarLabel::create( "p.sp_vol",
			ParticleVariable<double>::getTypeDescription() ); 
  
  pExternalHeatRateLabel = VarLabel::create( "p.externalHeatRate",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = VarLabel::create( "p.surface",
			ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pSizeLabel = VarLabel::create( "p.size",
			ParticleVariable<Vector>::getTypeDescription());
  
  pTang1Label  = VarLabel::create( "p.tang1",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label  = VarLabel::create( "p.tang2",
                        ParticleVariable<Vector>::getTypeDescription() );

  pNormLabel  = VarLabel::create( "p.norm",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang1Label_preReloc  = VarLabel::create( "p.tang1+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTang2Label_preReloc  = VarLabel::create( "p.tang2+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pNormLabel_preReloc  = VarLabel::create( "p.norm+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pSizeLabel_preReloc = VarLabel::create( "p.size+",
			ParticleVariable<Vector>::getTypeDescription());
  
  // Particle Variables 
  pDeformationMeasureLabel_preReloc = VarLabel::create("p.deformationMeasure+",
			ParticleVariable<Matrix3>::getTypeDescription());
  
  pStressLabel_preReloc = VarLabel::create( "p.stress+",
			ParticleVariable<Matrix3>::getTypeDescription() );

  pStress_eLabel = VarLabel::create( "p.stress_e",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pStress_veLabel = VarLabel::create( "p.stress_ve",
			ParticleVariable<Matrix3>::getTypeDescription() );
  
  pStress_ve_vLabel = VarLabel::create( "p.stress_ve_v",
			ParticleVariable<double>::getTypeDescription() );
			
  pStress_ve_dLabel = VarLabel::create( "p.stress_ve_d",
			ParticleVariable<Matrix3>::getTypeDescription() );
			
  pStress_e_vLabel = VarLabel::create( "p.stress_e_v",
			ParticleVariable<double>::getTypeDescription() );
			
  pStress_e_dLabel = VarLabel::create( "p.stress_e_d",
			ParticleVariable<Matrix3>::getTypeDescription() );												

  pCrackRadiusLabel_preReloc = VarLabel::create( "p.CrackRadius+",
			ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel_preReloc = VarLabel::create( "p.volume+",
			ParticleVariable<double>::getTypeDescription());
  
  pMassLabel_preReloc = VarLabel::create( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel_preReloc = VarLabel::create( "p.velocity+", 
			ParticleVariable<Vector>::getTypeDescription() );
  
  pExtForceLabel_preReloc = VarLabel::create( "p.externalforce+",
			ParticleVariable<Vector>::getTypeDescription() );
  
  pXLabel_preReloc = VarLabel::create( "p.x+",
			ParticleVariable<Point>::getTypeDescription(),
			IntVector(0,0,0),
			VarLabel::PositionVariable);
  
  pTemperatureLabel_preReloc = VarLabel::create( "p.temperature+",
			ParticleVariable<double>::getTypeDescription() );

  pSp_volLabel_preReloc = VarLabel::create( "p.Sp_vol+",
			ParticleVariable<double>::getTypeDescription() );  
                     
  pExternalHeatRateLabel_preReloc = VarLabel::create( "p.externalHeatRate+",
			ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = VarLabel::create( "p.surface+",
			ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

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
  
  gContactForceLabel  = VarLabel::create( "g.contactforce",
			NCVariable<Vector>::getTypeDescription() );
  
  gContactLabel       = VarLabel::create( "g.contact",
			NCVariable<int>::getTypeDescription() );
  
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

  gradPAccNCLabel = VarLabel::create("gradPAccNC",
			NCVariable<Vector>::getTypeDescription());

  dTdt_NCLabel     = VarLabel::create("dTdt_NC",
			NCVariable<double>::getTypeDescription());

  massBurnFractionLabel  = VarLabel::create("massBurnFraction",
			NCVariable<double>::getTypeDescription());
                     
  gSp_volLabel     =  VarLabel::create("g.sp_vol",
			NCVariable<double>::getTypeDescription()); 
 
  gSp_vol_srcLabel =  VarLabel::create("g.sp_vol_src",
			NCVariable<double>::getTypeDescription());

  // Interaction with Arches, Fluid Mechanics

  AccArchesNCLabel = VarLabel::create("AccArchesNC",
			NCVariable<Vector>::getTypeDescription() );

  // Interaction with Arches, Heat Transfer

  heaTranSolid_NCLabel = VarLabel::create("heaTranSolid_NC",
                                         NCVariable<double>::getTypeDescription() );

  frictionalWorkLabel = VarLabel::create("frictionalWork",
			NCVariable<double>::getTypeDescription());

  gNumNearParticlesLabel = VarLabel::create("NumNearParticles",
			NCVariable<double>::getTypeDescription());

  // Reduction variables
  partCountLabel = VarLabel::create("particleCount",
				   sumlong_vartype::getTypeDescription());

  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );

  StrainEnergyLabel = VarLabel::create( "StrainEnergy",
			sum_vartype::getTypeDescription() );

  AccStrainEnergyLabel = VarLabel::create( "AccStrainEnergy",
			max_vartype::getTypeDescription() );

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

  doMechLabel = VarLabel::create( "doMech", delt_vartype::getTypeDescription());

  // Implicit MPM labels

  gVelocityOldLabel = VarLabel::create("g.VelocityOld",
				    NCVariable<Vector>::getTypeDescription()); 

  pVolumeOldLabel = VarLabel::create("p.VolumeOld",
			       ParticleVariable<double>::getTypeDescription()); 

  dispNewLabel = VarLabel::create("dispNew",
				  NCVariable<Vector>::getTypeDescription()); 

  dispIncLabel = VarLabel::create("dispInc",
				  NCVariable<Vector>::getTypeDescription()); 

  dispIncQNorm0 = VarLabel::create("dispIncQNorm0",
				   sum_vartype::getTypeDescription());

  dispIncNormMax = VarLabel::create("dispIncNormMax",
				    sum_vartype::getTypeDescription());

  dispIncQNorm = VarLabel::create("dispIncQNorm",
				  sum_vartype::getTypeDescription());

  dispIncNorm = VarLabel::create("dispIncNorm",
				 sum_vartype::getTypeDescription());
  
  pAccelerationLabel = VarLabel::create("p.acceleration",
				   ParticleVariable<Vector>::getTypeDescription()); 

  pAccelerationLabel_preReloc = VarLabel::create("p.acceleration+",
				   ParticleVariable<Vector>::getTypeDescription()); 

  bElBarLabel = VarLabel::create("bElBar",
		       ParticleVariable<Matrix3>::getTypeDescription());

  bElBarLabel_preReloc = VarLabel::create("bElBar+",
			 ParticleVariable<Matrix3>::getTypeDescription());

  // for Fracture ----------------------------
  pDispLabel = VarLabel::create("p.displacement",
	          ParticleVariable<Vector>::getTypeDescription());
  pDispLabel_preReloc = VarLabel::create( "p.displacement+",
                  ParticleVariable<Vector>::getTypeDescription());

  pDispGradsLabel = VarLabel::create("p.dispGrads",
                  ParticleVariable<Matrix3>::getTypeDescription());
  pDispGradsLabel_preReloc = VarLabel::create( "p.dispGrads+",
                  ParticleVariable<Matrix3>::getTypeDescription());

  pStrainEnergyDensityLabel = VarLabel::create("p.strainEnergyDensity",
                  ParticleVariable<double>::getTypeDescription());
  pStrainEnergyDensityLabel_preReloc = VarLabel::create( "p.strainEnergyDensity+",
                  ParticleVariable<double>::getTypeDescription());

  pKineticEnergyDensityLabel = VarLabel::create("p.kineticEnergyDensity",
                  ParticleVariable<double>::getTypeDescription());

  pgCodeLabel = VarLabel::create("p.gcode",
                  ParticleVariable<Short27>::getTypeDescription());

  gNumPatlsLabel = VarLabel::create("g.numPatls",
                  NCVariable<int>::getTypeDescription());

  GNumPatlsLabel = VarLabel::create("G.numPatls",
                  NCVariable<int>::getTypeDescription());

  gDisplacementLabel = VarLabel::create("g.displacement",
                  NCVariable<Vector>::getTypeDescription() );

  GDisplacementLabel = VarLabel::create("G.displacement",
                  NCVariable<Vector>::getTypeDescription() );

  gGridStressLabel = VarLabel::create("g.gridStress",
                  NCVariable<Matrix3>::getTypeDescription());
  GGridStressLabel = VarLabel::create("G.gridStress",
                  NCVariable<Matrix3>::getTypeDescription());

  gDispGradsLabel = VarLabel::create("g.dispGrads",
                  NCVariable<Matrix3>::getTypeDescription());
  GDispGradsLabel = VarLabel::create("G.dispGrads",
                  NCVariable<Matrix3>::getTypeDescription());

  gStrainEnergyDensityLabel = VarLabel::create("g.strainEnergyDensity",
                  NCVariable<double>::getTypeDescription());
  GStrainEnergyDensityLabel = VarLabel::create("G.strainEnergyDensity",
                  NCVariable<double>::getTypeDescription());

  gKineticEnergyDensityLabel = VarLabel::create("g.kineticEnergyDensity",
                  NCVariable<double>::getTypeDescription());
  GKineticEnergyDensityLabel = VarLabel::create("G.kineticEnergyDensity",
                  NCVariable<double>::getTypeDescription());

  GCrackNormLabel= VarLabel::create("G.cracknormal",
                  NCVariable<Vector>::getTypeDescription());

  GMassLabel = VarLabel::create("G.mass",
                  NCVariable<double>::getTypeDescription());

  GVolumeLabel = VarLabel::create("G.volume",
                  NCVariable<double>::getTypeDescription());

  GVelocityLabel = VarLabel::create("G.velocity",
                  NCVariable<Vector>::getTypeDescription());

  GTemperatureLabel = VarLabel::create("G.temperature",
                  NCVariable<double>::getTypeDescription());

  GTemperatureNoBCLabel = VarLabel::create("G.temperatureiNoBC",
                  NCVariable<double>::getTypeDescription());

  GExternalForceLabel = VarLabel::create("G.externalforce",
                  NCVariable<Vector>::getTypeDescription());

  GExternalHeatRateLabel = VarLabel::create("G.externalheatrate",
                  NCVariable<double>::getTypeDescription());

  GThermalContactHeatExchangeRateLabel = VarLabel::create(
                  "G.thermalContactHeatExchangeRate",
                  NCVariable<double>::getTypeDescription());

  GInternalForceLabel = VarLabel::create("G.internalforce",
                  NCVariable<Vector>::getTypeDescription());

  GInternalHeatRateLabel = VarLabel::create("G.internalHeatRate",
                  NCVariable<double>::getTypeDescription());   

  GTemperatureRateLabel = VarLabel::create("G.temperatureRate",
                  NCVariable<double>::getTypeDescription());
     
  GTemperatureStarLabel = VarLabel::create("G.temperatureStar",
                  NCVariable<double>::getTypeDescription());
                   
  GVelocityStarLabel = VarLabel::create("G.velocityg_star",
                  NCVariable<Vector>::getTypeDescription() );

  GAccelerationLabel = VarLabel::create( "G.acceleration",
                  NCVariable<Vector>::getTypeDescription() );

  GSp_volLabel = VarLabel::create("G.sp_vol",
                  NCVariable<double>::getTypeDescription());

  GSp_vol_srcLabel =  VarLabel::create("G.sp_vol_src",
                  NCVariable<double>::getTypeDescription()); 
  // ------------------------------------------------------

  pVolumeOldLabel_preReloc = VarLabel::create("pVolumeOld+",
			       ParticleVariable<double>::getTypeDescription()); 

  // Material point erosion algorithms
  pErosionLabel = VarLabel::create("p.erosion",
                            ParticleVariable<double>::getTypeDescription());
  pErosionLabel_preReloc = VarLabel::create("p.erosion+",
                            ParticleVariable<double>::getTypeDescription());
  
  // MPM Physical BC labels (permanent particle state)
  materialPointsPerLoadCurveLabel = VarLabel::create("pointsPerCurve", 
                            sumlong_vartype::getTypeDescription());
  pLoadCurveIDLabel = VarLabel::create("p.loadCurveID",
                            ParticleVariable<int>::getTypeDescription());
  pLoadCurveIDLabel_preReloc = VarLabel::create("p.loadCurveID+",
                            ParticleVariable<int>::getTypeDescription());

  // MPM Artificial Damping labels (updated after each time step)
  pDampingRateLabel = VarLabel::create("dampingRate", 
                      sum_vartype::getTypeDescription());
  pDampingCoeffLabel = VarLabel::create("dampingCoeff", 
                       max_vartype::getTypeDescription());

  // Labels for shell materials
  pThickTopLabel = VarLabel::create("p.thickTop",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickTopLabel = VarLabel::create("p.thickTop0",
                     ParticleVariable<double>::getTypeDescription());
  pThickBotLabel = VarLabel::create("p.thickBot",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickBotLabel = VarLabel::create("p.thickBot0",
                     ParticleVariable<double>::getTypeDescription());
  pNormalLabel = VarLabel::create("p.normal",
                     ParticleVariable<Vector>::getTypeDescription());
  pInitialNormalLabel = VarLabel::create("p.normal0",
                     ParticleVariable<Vector>::getTypeDescription());

  pThickTopLabel_preReloc = VarLabel::create("p.thickTop+",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickTopLabel_preReloc = VarLabel::create("p.thickTop0+",
                     ParticleVariable<double>::getTypeDescription());
  pThickBotLabel_preReloc = VarLabel::create("p.thickBot+",
                     ParticleVariable<double>::getTypeDescription());
  pInitialThickBotLabel_preReloc = VarLabel::create("p.thickBot0+",
                     ParticleVariable<double>::getTypeDescription());
  pNormalLabel_preReloc = VarLabel::create("p.normal+",
                     ParticleVariable<Vector>::getTypeDescription());
  pInitialNormalLabel_preReloc = VarLabel::create("p.normal0+",
                     ParticleVariable<Vector>::getTypeDescription());

  gNormalRotRateLabel = VarLabel::create( "g.normalRotRate",
		     NCVariable<Vector>::getTypeDescription());
  gNormalRotMomentLabel = VarLabel::create( "g.normalRotMoment",
		     NCVariable<Vector>::getTypeDescription());

  gNormalRotMassLabel = VarLabel::create( "g.normalRotMass",
		     NCVariable<double>::getTypeDescription());
  gNormalRotAccLabel = VarLabel::create( "g.normalRotAcc",
		     NCVariable<Vector>::getTypeDescription());


} 

MPMLabel::~MPMLabel()
{
  //non PermanentParticleState
  VarLabel::destroy(pVolumeDeformedLabel);
  VarLabel::destroy(pTemperatureGradientLabel);
  VarLabel::destroy(p_qLabel);

  //PermanentParticleState
  VarLabel::destroy(pDeformationMeasureLabel);
  VarLabel::destroy(pDeformationMeasureLabel_preReloc);
  VarLabel::destroy(pStressLabel);
  VarLabel::destroy(pStressLabel_preReloc);
  VarLabel::destroy(pStress_eLabel);
  VarLabel::destroy(pStress_veLabel);
  VarLabel::destroy(pStress_ve_vLabel);
  VarLabel::destroy(pStress_ve_dLabel);
  VarLabel::destroy(pStress_e_vLabel);
  VarLabel::destroy(pStress_e_dLabel);
  VarLabel::destroy(pVolumeLabel);
  VarLabel::destroy(pVolumeLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pVelocityLabel);
  VarLabel::destroy(pVelocityLabel_preReloc);
  VarLabel::destroy(pExternalForceLabel);
  VarLabel::destroy(pExtForceLabel_preReloc);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pTemperatureLabel);
  VarLabel::destroy(pTemperatureLabel_preReloc);
  VarLabel::destroy(pSp_volLabel); 
  VarLabel::destroy(pSp_volLabel_preReloc); 
  VarLabel::destroy(pExternalHeatRateLabel);
  VarLabel::destroy(pExternalHeatRateLabel_preReloc);
  VarLabel::destroy(pSurfLabel);
  VarLabel::destroy(pSurfLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
  VarLabel::destroy(pPressureLabel);
  VarLabel::destroy(pCrackRadiusLabel);
  VarLabel::destroy(pCrackRadiusLabel_preReloc);
  VarLabel::destroy(pSizeLabel);
  VarLabel::destroy(pSizeLabel_preReloc);
  VarLabel::destroy(pTang1Label);
  VarLabel::destroy(pTang1Label_preReloc);
  VarLabel::destroy(pTang2Label);
  VarLabel::destroy(pTang2Label_preReloc);
  VarLabel::destroy(pNormLabel);
  VarLabel::destroy(pNormLabel_preReloc);

  VarLabel::destroy(gAccelerationLabel);
  VarLabel::destroy(gMomExedAccelerationLabel);
  VarLabel::destroy(gMassLabel);
  VarLabel::destroy(gVelocityLabel);
  VarLabel::destroy(gMomExedVelocityLabel);
  VarLabel::destroy(gExternalForceLabel);
  VarLabel::destroy(gInternalForceLabel);
  VarLabel::destroy(gContactForceLabel);
  VarLabel::destroy(gContactLabel);
  VarLabel::destroy(gVelocityStarLabel);
  VarLabel::destroy(gMomExedVelocityStarLabel);
  VarLabel::destroy(gNormTractionLabel);
  VarLabel::destroy(gStressLabel);
  VarLabel::destroy(gSurfNormLabel);
  VarLabel::destroy(gTemperatureLabel);
  VarLabel::destroy(gSp_volLabel); 
  VarLabel::destroy(gSp_vol_srcLabel); 
  VarLabel::destroy(gTemperatureNoBCLabel);
  VarLabel::destroy(gTemperatureStarLabel);
  VarLabel::destroy(gTemperatureRateLabel);
  VarLabel::destroy(gInternalHeatRateLabel);
  VarLabel::destroy(gExternalHeatRateLabel);
  VarLabel::destroy(gThermalContactHeatExchangeRateLabel);
  VarLabel::destroy(gStressForSavingLabel);
  VarLabel::destroy(gVolumeLabel);
  VarLabel::destroy(gradPAccNCLabel);
  VarLabel::destroy(dTdt_NCLabel);
  VarLabel::destroy(massBurnFractionLabel);
  VarLabel::destroy(AccArchesNCLabel);
  VarLabel::destroy(heaTranSolid_NCLabel);
  VarLabel::destroy(frictionalWorkLabel);
  VarLabel::destroy(gNumNearParticlesLabel);

  VarLabel::destroy(partCountLabel);
  VarLabel::destroy(delTLabel);
  VarLabel::destroy(doMechLabel);

  VarLabel::destroy(AccStrainEnergyLabel);
  VarLabel::destroy(StrainEnergyLabel);
  VarLabel::destroy(KineticEnergyLabel);
  VarLabel::destroy(TotalMassLabel);
  VarLabel::destroy(NTractionZMinusLabel);
  VarLabel::destroy(CenterOfMassPositionLabel);
  VarLabel::destroy(CenterOfMassVelocityLabel);
  VarLabel::destroy(pCellNAPIDLabel);

  VarLabel::destroy(gVelocityOldLabel);
  VarLabel::destroy(pVolumeOldLabel);
  VarLabel::destroy(dispNewLabel);
  VarLabel::destroy(dispIncLabel);
  VarLabel::destroy(pAccelerationLabel);
  VarLabel::destroy(dispIncQNorm0);
  VarLabel::destroy(dispIncNormMax);
  VarLabel::destroy(dispIncQNorm);
  VarLabel::destroy(dispIncNorm);
  VarLabel::destroy(pAccelerationLabel_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
  VarLabel::destroy(pVolumeOldLabel_preReloc);

 // for Fracture --------------
  VarLabel::destroy(pDispLabel);
  VarLabel::destroy(pDispLabel_preReloc);
  VarLabel::destroy(pDispGradsLabel);
  VarLabel::destroy(pDispGradsLabel_preReloc);
  VarLabel::destroy(pStrainEnergyDensityLabel);
  VarLabel::destroy(pStrainEnergyDensityLabel_preReloc);
  VarLabel::destroy(pKineticEnergyDensityLabel);

  VarLabel::destroy(pgCodeLabel);

  VarLabel::destroy(gNumPatlsLabel);
  VarLabel::destroy(GNumPatlsLabel);
  VarLabel::destroy(gDisplacementLabel);
  VarLabel::destroy(GDisplacementLabel);
  VarLabel::destroy(gGridStressLabel);
  VarLabel::destroy(GGridStressLabel);
  VarLabel::destroy(gDispGradsLabel);
  VarLabel::destroy(GDispGradsLabel);
  VarLabel::destroy(gStrainEnergyDensityLabel);
  VarLabel::destroy(GStrainEnergyDensityLabel);
  VarLabel::destroy(gKineticEnergyDensityLabel);
  VarLabel::destroy(GKineticEnergyDensityLabel);

  VarLabel::destroy(GCrackNormLabel);
  VarLabel::destroy(GMassLabel);
  VarLabel::destroy(GVolumeLabel);
  VarLabel::destroy(GVelocityLabel);
  VarLabel::destroy(GTemperatureLabel);
  VarLabel::destroy(GTemperatureNoBCLabel);
  VarLabel::destroy(GExternalForceLabel);
  VarLabel::destroy(GExternalHeatRateLabel);
  VarLabel::destroy(GThermalContactHeatExchangeRateLabel);
  VarLabel::destroy(GInternalForceLabel);
  VarLabel::destroy(GInternalHeatRateLabel);
  VarLabel::destroy(GTemperatureRateLabel);
  VarLabel::destroy(GTemperatureStarLabel);
  VarLabel::destroy(GVelocityStarLabel);
  VarLabel::destroy(GAccelerationLabel);
  VarLabel::destroy(GSp_volLabel);
  VarLabel::destroy(GSp_vol_srcLabel);
  // --------------------------------

  // Destroy Material point erosion labels
  VarLabel::destroy(pErosionLabel);
  VarLabel::destroy(pErosionLabel_preReloc);

  // Destroy the MPM Physical BC pointer labels
  VarLabel::destroy(materialPointsPerLoadCurveLabel);
  VarLabel::destroy(pLoadCurveIDLabel);
  VarLabel::destroy(pLoadCurveIDLabel_preReloc);

  // Destroy the MPM Damping rate labels
  VarLabel::destroy(pDampingRateLabel);
  VarLabel::destroy(pDampingCoeffLabel);

  // Destroy Labels for shell materials
  VarLabel::destroy(pThickTopLabel);
  VarLabel::destroy(pInitialThickTopLabel);
  VarLabel::destroy(pThickBotLabel);
  VarLabel::destroy(pInitialThickBotLabel);
  VarLabel::destroy(pNormalLabel);
  VarLabel::destroy(pInitialNormalLabel);

  VarLabel::destroy(pThickTopLabel_preReloc);
  VarLabel::destroy(pInitialThickTopLabel_preReloc);
  VarLabel::destroy(pThickBotLabel_preReloc);
  VarLabel::destroy(pInitialThickBotLabel_preReloc);
  VarLabel::destroy(pNormalLabel_preReloc);
  VarLabel::destroy(pInitialNormalLabel_preReloc);

  VarLabel::destroy(gNormalRotRateLabel); 
  VarLabel::destroy(gNormalRotMomentLabel); 
  VarLabel::destroy(gNormalRotMassLabel); 
  VarLabel::destroy(gNormalRotAccLabel); 
}
