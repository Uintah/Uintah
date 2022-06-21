/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using namespace std;

MPMLabel::MPMLabel()
{

  // Time Step
  timeStepLabel =
    VarLabel::create(timeStep_name, timeStep_vartype::getTypeDescription() );

  // Simulation Time
  simulationTimeLabel =
    VarLabel::create(simTime_name, simTime_vartype::getTypeDescription() );

  // delta t
  VarLabel* nonconstDelt =
    VarLabel::create(delT_name, delt_vartype::getTypeDescription() );
  nonconstDelt->allowMultipleComputes();
  delTLabel = nonconstDelt;


  diffusion = scinew MPMDiffusionLabel();

  // Heat flux from fire

  fAllParticlesUpdated = VarLabel::create("finalUpdateFlag",
                                          PerPatch<bool>::getTypeDescription());

  heatRate_CCLabel = 
    VarLabel::create("heatRate_CC",  CCVariable<double>::getTypeDescription());

  // Particle Variables

  //non PermanentParticleState
  pPressureLabel  = VarLabel::create( "p.pressure",
                        ParticleVariable<double>::getTypeDescription() );
  
  pScratchVecLabel  = VarLabel::create( "p.scratchvec",
                        ParticleVariable<Vector>::getTypeDescription() );
  
  pScratchLabel  = VarLabel::create( "p.scratch",
                        ParticleVariable<double>::getTypeDescription() );
  
  // for visualization only
  pScaleFactorLabel = VarLabel::create( "p.scalefactor",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  pScaleFactorLabel_preReloc = VarLabel::create( "p.scalefactor+",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  // for thermal stress
  pTempCurrentLabel = VarLabel::create( "p.tempCurrent",
                        ParticleVariable<double>::getTypeDescription() ); 
    
  pXXLabel  = VarLabel::create( "p.xx",
                        ParticleVariable<Point>::getTypeDescription() );
                     
  p_qLabel  = VarLabel::create( "p.q",
                        ParticleVariable<double>::getTypeDescription() );
  
  p_qLabel_preReloc  = VarLabel::create( "p.q+",
                        ParticleVariable<double>::getTypeDescription() );
  
  pColorLabel  = VarLabel::create( "p.color",
                        ParticleVariable<double>::getTypeDescription() );
                     
  pColorLabel_preReloc  = VarLabel::create( "p.color+",
                        ParticleVariable<double>::getTypeDescription() );
                     
  pPartitionUnityLabel   = VarLabel::create( "p.partitionUnity",
                        ParticleVariable<double>::getTypeDescription() );
  
  //PermanentParticleState
  pDeformationMeasureLabel = VarLabel::create("p.deformationMeasure",
                        ParticleVariable<Matrix3>::getTypeDescription());

  pVelGradLabel = VarLabel::create("p.velGrad",
                        ParticleVariable<Matrix3>::getTypeDescription());

  pStressLabel = VarLabel::create( "p.stress",
                        ParticleVariable<Matrix3>::getTypeDescription() );
  
  pVolumeLabel = VarLabel::create( "p.volume",
                        ParticleVariable<double>::getTypeDescription());
  
  pVolumeDeformedLabel = VarLabel::create( "p.volumedeformed",
                        ParticleVariable<double>::getTypeDescription());
  
  pMassLabel = VarLabel::create( "p.mass",
                        ParticleVariable<double>::getTypeDescription() );
  
  pVelocityLabel = VarLabel::create( "p.velocity", 
                        ParticleVariable<Vector>::getTypeDescription() );
  
  pVelocitySSPlusLabel = VarLabel::create( "p.velocitySSPlus", 
                        ParticleVariable<Vector>::getTypeDescription() );
  
  pExternalForceLabel = VarLabel::create( "p.externalforce",
                        ParticleVariable<Vector>::getTypeDescription() );

  pExternalForceCorner1Label = VarLabel::create( "p.externalforcecorner1",
                        ParticleVariable<Point>::getTypeDescription() );

  pExternalForceCorner2Label = VarLabel::create( "p.externalforcecorner2",
                        ParticleVariable<Point>::getTypeDescription() );

  pExternalForceCorner3Label = VarLabel::create( "p.externalforcecorner3",
                        ParticleVariable<Point>::getTypeDescription() );

  pExternalForceCorner4Label = VarLabel::create( "p.externalforcecorner4",
                        ParticleVariable<Point>::getTypeDescription() );
  
  pXLabel = VarLabel::create("p.x",
                             ParticleVariable<Point>::getTypeDescription(),
                             IntVector(0,0,0), VarLabel::PositionVariable);
  
  pTemperatureLabel = VarLabel::create( "p.temperature",
                        ParticleVariable<double>::getTypeDescription() );
  
  // for thermal stress
  pTempPreviousLabel = VarLabel::create( "p.tempPrevious",
                        ParticleVariable<double>::getTypeDescription() ); 
                     
  pdTdtLabel = VarLabel::create( "p.dTdt",
                        ParticleVariable<double>::getTypeDescription() );
  
  pExternalHeatRateLabel = VarLabel::create( "p.externalHeatRate",
                        ParticleVariable<double>::getTypeDescription() );

  pExternalHeatFluxLabel = VarLabel::create( "p.externalHeatFlux",
                        ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel = VarLabel::create( "p.surface",
                        ParticleVariable<double>::getTypeDescription() );

  pSurfGradLabel = VarLabel::create( "p.surfgrad",
                        ParticleVariable<Vector>::getTypeDescription() );

  pSurfGradLabel_preReloc = VarLabel::create( "p.surfgrad+",
                        ParticleVariable<Vector>::getTypeDescription() );

  pLastLevelLabel = VarLabel::create( "p.lastlevel",
                        ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
                        ParticleVariable<long64>::getTypeDescription() );

  pSizeLabel = VarLabel::create( "p.size",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  pSizeLabel_preReloc = VarLabel::create( "p.size+",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  pCurSizeLabel = VarLabel::create( "p.currentsize",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  pLocalizedMPMLabel  = VarLabel::create( "p.localizedMPM",
                        ParticleVariable<int>::getTypeDescription() );
  
  pLocalizedMPMLabel_preReloc  = VarLabel::create( "p.localizedMPM+",
                        ParticleVariable<int>::getTypeDescription() );
  
  pRefinedLabel       = VarLabel::create( "p.refinedMPM",
                        ParticleVariable<int>::getTypeDescription() );
  
  pRefinedLabel_preReloc  = VarLabel::create( "p.refinedMPM+",
                        ParticleVariable<int>::getTypeDescription() );

  pESPotential = VarLabel::create( "p.esPotential",
      ParticleVariable<double>::getTypeDescription());

  pESGradPotential = VarLabel::create( "p.esGradPotential",
                     ParticleVariable<Vector>::getTypeDescription() );

  pFiberDirLabel  = VarLabel::create( "p.fiberdir",
                        ParticleVariable<Vector>::getTypeDescription() );

  pFiberDirLabel_preReloc  = VarLabel::create( "p.fiberdir+",
                        ParticleVariable<Vector>::getTypeDescription() );

  // Particle Variables 
  pDeformationMeasureLabel_preReloc = VarLabel::create("p.deformationMeasure+",
                        ParticleVariable<Matrix3>::getTypeDescription());
  
  pVelGradLabel_preReloc = VarLabel::create("p.velGrad+",
                        ParticleVariable<Matrix3>::getTypeDescription());

  pStressLabel_preReloc = VarLabel::create( "p.stress+",
                        ParticleVariable<Matrix3>::getTypeDescription() );

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
    
  // for thermal stress
  pTempPreviousLabel_preReloc = VarLabel::create( "p.tempPrevious+",
                        ParticleVariable<double>::getTypeDescription() ); 

  pdTdtLabel_preReloc = VarLabel::create( "p.dTdt+",
                        ParticleVariable<double>::getTypeDescription() );

  pExternalHeatRateLabel_preReloc = VarLabel::create( "p.externalHeatRate+",
                        ParticleVariable<double>::getTypeDescription() );

  pExternalHeatFluxLabel_preReloc = VarLabel::create( "p.externalHeatFlux+",
                        ParticleVariable<double>::getTypeDescription() );
  
  pSurfLabel_preReloc = VarLabel::create( "p.surface+",
                        ParticleVariable<double>::getTypeDescription() );

  pLastLevelLabel_preReloc = VarLabel::create( "p.lastlevel+",
                        ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
                        ParticleVariable<long64>::getTypeDescription() );

  pTemperatureGradientLabel = VarLabel::create( "p.temperatureGradient",
                        ParticleVariable<Vector>::getTypeDescription() );

  pTemperatureGradientLabel_preReloc =VarLabel::create("p.temperatureGradient+",
                        ParticleVariable<Vector>::getTypeDescription() );

  // Node Centered Variables
  
  gAccelerationLabel = VarLabel::create( "g.acceleration",
                        NCVariable<Vector>::getTypeDescription() );

  gColorLabel = VarLabel::create( "g.color",
                        NCVariable<double>::getTypeDescription() );
  
  gMatlsOnNodeLabel = VarLabel::create( "g.matlsOnNode",
                        NCVariable<IntVector>::getTypeDescription() );
  
  gMassLabel = VarLabel::create( "g.mass",
                        NCVariable<double>::getTypeDescription() );
  
  gMassAllLabel = VarLabel::create( "g.massall",
                        NCVariable<double>::getTypeDescription() );
  
  gMassF0Label = VarLabel::create( "g.massF0",
                        NCVariable<double>::getTypeDescription() );

  gMassF1Label = VarLabel::create( "g.massF1",
                        NCVariable<double>::getTypeDescription() );

  gVelocityLabel = VarLabel::create( "g.velocity",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelocityF0Label = VarLabel::create( "g.velocityF0",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelocityF1Label = VarLabel::create( "g.velocityF1",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarF0Label = VarLabel::create( "g.velocity_starF0",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelocityStarF1Label = VarLabel::create( "g.velocity_starF1",
                        NCVariable<Vector>::getTypeDescription() );
  
  gAccelerationF0Label = VarLabel::create( "g.accelerationF0",
                        NCVariable<Vector>::getTypeDescription() );
  
  gAccelerationF1Label = VarLabel::create( "g.accelerationF1",
                        NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceF0Label = VarLabel::create( "g.internalforceF0",
                        NCVariable<Vector>::getTypeDescription() );
  
  gInternalForceF1Label = VarLabel::create( "g.internalforceF1",
                        NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceF0Label = VarLabel::create( "g.externalforceF0",
                        NCVariable<Vector>::getTypeDescription() );
  
  gExternalForceF1Label = VarLabel::create( "g.externalforceF1",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelocityBCLabel = VarLabel::create( "g.velocityBC",
                        NCVariable<Vector>::getTypeDescription() );
  
  gVelSPSSPLabel = VarLabel::create( "g.velocitySPLusSSPlus",
                        NCVariable<Vector>::getTypeDescription() );
  
  gMatlProminenceLabel = VarLabel::create( "g.matlProminence",
                        NCVariable<double>::getTypeDescription() );
  
  gAlphaMaterialLabel = VarLabel::create( "g.alphaMaterial",
                        NCVariable<int>::getTypeDescription() );
  
  gNormAlphaToBetaLabel = VarLabel::create( "g.normAlphaToBeta",
                        NCVariable<Vector>::getTypeDescription() );
  
  gPositionLabel = VarLabel::create( "g.position",
                        NCVariable<Point>::getTypeDescription() );
  
  gPositionF0Label = VarLabel::create( "g.positionF0",
                        NCVariable<Point>::getTypeDescription() );
  
  gPositionF1Label = VarLabel::create( "g.positionF1",
                        NCVariable<Point>::getTypeDescription() );
  
  gExternalForceLabel = VarLabel::create( "g.externalforce",
                        NCVariable<Vector>::getTypeDescription() );
  
  gLSContactForceLabel = VarLabel::create( "g.lscontactforce",
                        NCVariable<Vector>::getTypeDescription() );
  
  gInContactMatlLabel = VarLabel::create( "g.incontactmatl",
                        NCVariable<int>::getTypeDescription() );
  
  gInternalForceLabel = VarLabel::create( "g.internalforce",
                        NCVariable<Vector>::getTypeDescription() );
  
  gContactLabel       = VarLabel::create( "g.contact",
                        NCVariable<int>::getTypeDescription() );
  
  gVelocityStarLabel = VarLabel::create( "g.velocity_star",
                        NCVariable<Vector>::getTypeDescription() );
  
  gTemperatureLabel = VarLabel::create("g.temperature",
                        NCVariable<double>::getTypeDescription());

  gTemperatureNoBCLabel = VarLabel::create("g.temperaturenobc",
                        NCVariable<double>::getTypeDescription());

  gTemperatureStarLabel = VarLabel::create("g.temperatureStar",
                        NCVariable<double>::getTypeDescription());

  gTemperatureRateLabel = VarLabel::create("g.temperatureRate",
                        NCVariable<double>::getTypeDescription());

  gdTdtLabel = VarLabel::create("g.dTdt",
                        NCVariable<double>::getTypeDescription());

  gHeatFluxLabel = VarLabel::create("g.HeatFlux",
                        NCVariable<Vector>::getTypeDescription());

  gExternalHeatRateLabel = VarLabel::create("g.externalHeatRate",
                        NCVariable<double>::getTypeDescription());

  gExternalHeatFluxLabel = VarLabel::create("g.externalHeatFlux",
                       NCVariable<double>::getTypeDescription());

  NC_CCweightLabel     = VarLabel::create("NC_CCweight",
                     NCVariable<double>::getTypeDescription());

  gThermalContactTemperatureRateLabel = 
     VarLabel::create("g.thermalContactTemperatureRate",
     NCVariable<double>::getTypeDescription());

  gSurfaceAreaLabel = VarLabel::create( "g.surfacearea",
                   NCVariable<double>::getTypeDescription() );

  gSurfaceClayLabel = VarLabel::create( "g.surfaceclay",
                   NCVariable<double>::getTypeDescription() );

  gSurfaceCementLabel = VarLabel::create( "g.surfacecement",
                   NCVariable<double>::getTypeDescription() );

  gNormTractionLabel = VarLabel::create( "g.normtraction",
                   NCVariable<double>::getTypeDescription() );

  gNormTractionF0Label = VarLabel::create( "g.normtractionF0",
                   NCVariable<double>::getTypeDescription() );

  gNormTractionF1Label = VarLabel::create( "g.normtractionF1",
                   NCVariable<double>::getTypeDescription() );

  gSurfNormLabel = VarLabel::create( "g.surfnorm",
                   NCVariable<Vector>::getTypeDescription() );

  gSurfNormF0Label = VarLabel::create( "g.surfnormF0",
                   NCVariable<Vector>::getTypeDescription() );

  gSurfNormF1Label = VarLabel::create( "g.surfnormF1",
                   NCVariable<Vector>::getTypeDescription() );

  gSurfLabel     = VarLabel::create( "g.surf",
                   NCVariable<double>::getTypeDescription() );

  gSurfGradLabel = VarLabel::create( "g.surfgrad",
                   NCVariable<Vector>::getTypeDescription() );

  gStressLabel   = VarLabel::create( "g.stress",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressF0Label   = VarLabel::create( "g.stressF0",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressF1Label   = VarLabel::create( "g.stressF1",
                   NCVariable<Matrix3>::getTypeDescription() );

  gStressForSavingLabel   = VarLabel::create( "g.stressFS",
                   NCVariable<Matrix3>::getTypeDescription() );

  gVolumeLabel     = VarLabel::create("g.volume",
                        NCVariable<double>::getTypeDescription());

  gVolumeF0Label   = VarLabel::create("g.volumeF0",
                        NCVariable<double>::getTypeDescription());

  gVolumeF1Label   = VarLabel::create("g.volumeF1",
                        NCVariable<double>::getTypeDescription());

  gZOILabel     = VarLabel::create("g.zoi",
                        NCVariable<Stencil7>::getTypeDescription());

  MPMRefineCellLabel  = VarLabel::create( "MPMRefineCell",
                     CCVariable<double>::getTypeDescription() );

  cVolumeLabel  = VarLabel::create( "c.volume",
                     CCVariable<double>::getTypeDescription() );

  numLocInCellLabel  = VarLabel::create( "NumLocalizedInCell",
                     CCVariable<int>::getTypeDescription() );

  numInCellLabel  = VarLabel::create( "NumInCell",
                     CCVariable<int>::getTypeDescription() );

  TotalVolumeDeformedLabel = VarLabel::create( "TotalVolumeDeformed",
                                 sum_vartype::getTypeDescription() );
  
  gradPAccNCLabel = VarLabel::create("gradPAccNC",
                        NCVariable<Vector>::getTypeDescription());

  dTdt_NCLabel     = VarLabel::create("dTdt_NC",
                        NCVariable<double>::getTypeDescription());

  massBurnFractionLabel  = VarLabel::create("massBurnFraction",
                        NCVariable<double>::getTypeDescription());
                     
  dLdtDissolutionLabel  = VarLabel::create("dLdtDissolution",
                        NCVariable<double>::getTypeDescription());
                     
  gCemVecLabel  = VarLabel::create("g.CemVec",
                        NCVariable<Vector>::getTypeDescription());
                     
  NodalWeightSumLabel  = VarLabel::create("NodalWeightSum",
                        NCVariable<double>::getTypeDescription());

  gSp_volLabel     =  VarLabel::create("g.sp_vol",
                        NCVariable<double>::getTypeDescription()); 
 
  gSp_vol_srcLabel =  VarLabel::create("g.sp_vol_src",
                        NCVariable<double>::getTypeDescription());

  frictionalWorkLabel = VarLabel::create("frictionalWork",
                        NCVariable<double>::getTypeDescription());

  gNumNearParticlesLabel = VarLabel::create("NumNearParticles",
                        NCVariable<double>::getTypeDescription());

  // Reduction variables
  partCountLabel = VarLabel::create("particleCount",
                                   sumlong_vartype::getTypeDescription());

  czCountLabel = VarLabel::create("czCount",
                                   sumlong_vartype::getTypeDescription());

  lineSegmentCountLabel = VarLabel::create("lineSegmentCount",
                                   sumlong_vartype::getTypeDescription());

  TimeAveSpecificKELabel = VarLabel::create("TimeAveSpecificKE",
                                   max_vartype::getTypeDescription());

  StrainEnergyLabel = VarLabel::create( "StrainEnergy",
                        sum_vartype::getTypeDescription() );

  AccStrainEnergyLabel = VarLabel::create( "AccStrainEnergy",
                        max_vartype::getTypeDescription() );

  KineticEnergyLabel = VarLabel::create( "KineticEnergy",
                        sum_vartype::getTypeDescription() );

  ThermalEnergyLabel = VarLabel::create( "ThermalEnergy",
                        sum_vartype::getTypeDescription() );

  TotalMassLabel = VarLabel::create( "TotalMass",
                                 sum_vartype::getTypeDescription() );

  InitialMassSVLabel = VarLabel::create( "InitialMassSV",
                                 SoleVariable<double>::getTypeDescription());

  DissolvedMassLabel = VarLabel::create( "DissolvedMass",
                                 sum_vartype::getTypeDescription() );

  PistonMassLabel = VarLabel::create( "PistonMass",
                                 sum_vartype::getTypeDescription() );

  AddedParticlesLabel = VarLabel::create("AddedParticles",
                                 sum_vartype::getTypeDescription());

  TotalSurfaceAreaLabel = VarLabel::create( "TotalSurfaceArea",
                                 sum_vartype::getTypeDescription() );

  for(int iside=0;iside<6;iside++) {
      string label_name = Patch::getFaceName( (Patch::FaceType) iside ); // FIXME: assumes face indices
      
      BndyContactAreaLabel[iside] =
        VarLabel::create( std::string("BndyContactArea_"+label_name).c_str(),
                          sum_vartype::getTypeDescription() );
      BndyContactCellAreaLabel[iside] =
        VarLabel::create( std::string("BndyContactCellArea_"+label_name).c_str(),
                          sum_vartype::getTypeDescription() );
      BndyForceLabel[iside] =
        VarLabel::create( std::string("BndyForce_"+label_name).c_str(),
                          sumvec_vartype::getTypeDescription() );
      BndyTractionLabel[iside] =
        VarLabel::create( std::string("BndyTraction_"+label_name).c_str(),
                          sumvec_vartype::getTypeDescription() );
  }
  
  CenterOfMassPositionLabel = VarLabel::create( "CenterOfMassPosition",
                                 sumvec_vartype::getTypeDescription() );

  TotalMomentumLabel = VarLabel::create( "TotalMomentum",
                                 sumvec_vartype::getTypeDescription() );

  RigidReactionForceLabel = VarLabel::create( "RigidReactionForce",
                                 sumvec_vartype::getTypeDescription() );

  TotalLocalizedParticleLabel = VarLabel::create("TotalLocalizedParticle",
                                   sumlong_vartype::getTypeDescription());


  // for assigning particle ids
  pCellNAPIDLabel =
    VarLabel::create("cellNAPID", CCVariable<int>::getTypeDescription());

  doMechLabel = VarLabel::create( "doMech", delt_vartype::getTypeDescription());

  // Implicit MPM labels

  gVelocityOldLabel = VarLabel::create("g.VelocityOld",
                                    NCVariable<Vector>::getTypeDescription()); 

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

  // for Fracture ----------------------------
  pDispLabel = VarLabel::create("p.displacement",
                  ParticleVariable<Vector>::getTypeDescription());
  pDispLabel_preReloc = VarLabel::create( "p.displacement+",
                  ParticleVariable<Vector>::getTypeDescription());

  gDisplacementLabel = VarLabel::create("g.displacement",
                  NCVariable<Vector>::getTypeDescription() );

  // Material point erosion algorithms
  pErosionLabel = VarLabel::create("p.erosion",
                            ParticleVariable<double>::getTypeDescription());
  pErosionLabel_preReloc = VarLabel::create("p.erosion+",
                            ParticleVariable<double>::getTypeDescription());
  
  // MPM Physical BC labels (permanent particle state)
  materialPointsPerLoadCurveLabel = VarLabel::create("pointsPerCurve", 
                            sumlong_vartype::getTypeDescription());
  pLoadCurveIDLabel = VarLabel::create("p.loadCurveID",
                            ParticleVariable<IntVector>::getTypeDescription());
  pLoadCurveIDLabel_preReloc = VarLabel::create("p.loadCurveID+",
                            ParticleVariable<IntVector>::getTypeDescription());

  // For Cohesive Zones
  czAreaLabel = VarLabel::create("cz.area",
                     ParticleVariable<double>::getTypeDescription());
  czAreaLabel_preReloc = VarLabel::create("cz.area+",
                     ParticleVariable<double>::getTypeDescription());

  czNormLabel = VarLabel::create("cz.norm",
                     ParticleVariable<Vector>::getTypeDescription());
  czNormLabel_preReloc = VarLabel::create("cz.norm+",
                     ParticleVariable<Vector>::getTypeDescription());

  czTangLabel = VarLabel::create("cz.tang",
                     ParticleVariable<Vector>::getTypeDescription());
  czTangLabel_preReloc = VarLabel::create("cz.tang+",
                     ParticleVariable<Vector>::getTypeDescription());

  czDispTopLabel = VarLabel::create("cz.disptop",
                     ParticleVariable<Vector>::getTypeDescription());
  czDispTopLabel_preReloc = VarLabel::create("cz.disptop+",
                     ParticleVariable<Vector>::getTypeDescription());

  czDispBottomLabel = VarLabel::create("cz.dispbottom",
                     ParticleVariable<Vector>::getTypeDescription());
  czDispBottomLabel_preReloc = VarLabel::create("cz.dispbottom+",
                     ParticleVariable<Vector>::getTypeDescription());

  czSeparationLabel = VarLabel::create("cz.separation",
                     ParticleVariable<Vector>::getTypeDescription());
  czSeparationLabel_preReloc = VarLabel::create("cz.separation+",
                     ParticleVariable<Vector>::getTypeDescription());

  czForceLabel = VarLabel::create("cz.force",
                     ParticleVariable<Vector>::getTypeDescription());
  czForceLabel_preReloc = VarLabel::create("cz.force+",
                     ParticleVariable<Vector>::getTypeDescription());

  czTopMatLabel = VarLabel::create("cz.topmat",
                     ParticleVariable<int>::getTypeDescription());
  czTopMatLabel_preReloc = VarLabel::create("cz.topmat+",
                     ParticleVariable<int>::getTypeDescription());

  czBotMatLabel = VarLabel::create("cz.botmat",
                     ParticleVariable<int>::getTypeDescription());
  czBotMatLabel_preReloc = VarLabel::create("cz.botmat+",
                     ParticleVariable<int>::getTypeDescription());

  czFailedLabel = VarLabel::create("cz.failed",
                     ParticleVariable<int>::getTypeDescription());
  czFailedLabel_preReloc = VarLabel::create("cz.failed+",
                     ParticleVariable<int>::getTypeDescription());

  czIDLabel = VarLabel::create("cz.CZID",
                        ParticleVariable<long64>::getTypeDescription() );

  czIDLabel_preReloc = VarLabel::create("cz.CZID+",
                        ParticleVariable<long64>::getTypeDescription() );

  linesegIDLabel = VarLabel::create("ls.linesegID",
			ParticleVariable<long64>::getTypeDescription() );

  linesegIDLabel_preReloc = VarLabel::create("ls.linesegID+",
			ParticleVariable<long64>::getTypeDescription() );

  lsMidToEndVectorLabel = VarLabel::create("ls.MidToEndVector",
			ParticleVariable<Vector>::getTypeDescription() );

  lsMidToEndVectorLabel_preReloc = VarLabel::create("ls.MidToEndVector+",
			ParticleVariable<Vector>::getTypeDescription() );

  pModalIDLabel = VarLabel::create("p.modalID",
			ParticleVariable<int>::getTypeDescription() );

  pModalIDLabel_preReloc = VarLabel::create("p.modalID+",
			ParticleVariable<int>::getTypeDescription() );

  // for assigning particle ids
  pCellNACZIDLabel =
    VarLabel::create("cellNACZID", CCVariable<short int>::getTypeDescription());

  bElBarLabel                = VarLabel::create("p.bElBar",
                             ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc       = VarLabel::create("p.bElBar+",
                             ParticleVariable<Matrix3>::getTypeDescription());

  pPlasticStrainLabel          = VarLabel::create("p.plasticStrain", 
                         ParticleVariable<double>::getTypeDescription() );
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
                         ParticleVariable<double>::getTypeDescription() );
  pYieldStressLabel            = VarLabel::create("p.yieldStress",   
                         ParticleVariable<double>::getTypeDescription() );
  pYieldStressLabel_preReloc   = VarLabel::create("p.yieldStress+",  
                         ParticleVariable<double>::getTypeDescription() );
}

MPMLabel::~MPMLabel()
{
  if (diffusion) {
    delete diffusion;
  }

  VarLabel::destroy(timeStepLabel);
  VarLabel::destroy(simulationTimeLabel);
  VarLabel::destroy(delTLabel);

  VarLabel::destroy(fAllParticlesUpdated);

  VarLabel::destroy(heatRate_CCLabel);
  //non PermanentParticleState
  VarLabel::destroy(pVolumeDeformedLabel);
  VarLabel::destroy(pTempCurrentLabel); // for thermal stress
  VarLabel::destroy(pXXLabel);

  //PermanentParticleState
  VarLabel::destroy(pDeformationMeasureLabel);
  VarLabel::destroy(pDeformationMeasureLabel_preReloc);
  VarLabel::destroy(pVelGradLabel);
  VarLabel::destroy(pVelGradLabel_preReloc);
  VarLabel::destroy(pStressLabel);
  VarLabel::destroy(pStressLabel_preReloc);
  VarLabel::destroy(pVolumeLabel);
  VarLabel::destroy(pVolumeLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pVelocityLabel);
  VarLabel::destroy(pVelocityLabel_preReloc);
  VarLabel::destroy(pVelocitySSPlusLabel);
  VarLabel::destroy(pExternalForceLabel);
  VarLabel::destroy(pExternalForceCorner1Label);
  VarLabel::destroy(pExternalForceCorner2Label);
  VarLabel::destroy(pExternalForceCorner3Label);
  VarLabel::destroy(pExternalForceCorner4Label);
  VarLabel::destroy(pExtForceLabel_preReloc);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pTemperatureLabel);
  VarLabel::destroy(pTemperatureLabel_preReloc);
  VarLabel::destroy(pTempPreviousLabel); // for thermal stress
  VarLabel::destroy(pTempPreviousLabel_preReloc); // for thermal stress
  VarLabel::destroy(pdTdtLabel);
  VarLabel::destroy(pdTdtLabel_preReloc);
  VarLabel::destroy(pExternalHeatRateLabel);
  VarLabel::destroy(pExternalHeatRateLabel_preReloc);
  VarLabel::destroy(pExternalHeatFluxLabel);
  VarLabel::destroy(pExternalHeatFluxLabel_preReloc);
  VarLabel::destroy(pSurfLabel);
  VarLabel::destroy(pSurfLabel_preReloc);
  VarLabel::destroy(pSurfGradLabel);
  VarLabel::destroy(pSurfGradLabel_preReloc);
  VarLabel::destroy(pLastLevelLabel);
  VarLabel::destroy(pLastLevelLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
  VarLabel::destroy(czIDLabel);
  VarLabel::destroy(czIDLabel_preReloc);
  VarLabel::destroy(linesegIDLabel);
  VarLabel::destroy(linesegIDLabel_preReloc);
  VarLabel::destroy(lsMidToEndVectorLabel);
  VarLabel::destroy(lsMidToEndVectorLabel_preReloc);
  VarLabel::destroy(pModalIDLabel);
  VarLabel::destroy(pModalIDLabel_preReloc);
  VarLabel::destroy(pPressureLabel);
  VarLabel::destroy(pScratchVecLabel);
  VarLabel::destroy(pScratchLabel);
  VarLabel::destroy(pScaleFactorLabel);
  VarLabel::destroy(pScaleFactorLabel_preReloc);
  VarLabel::destroy(pLocalizedMPMLabel);
  VarLabel::destroy(pLocalizedMPMLabel_preReloc);
  VarLabel::destroy(pRefinedLabel);
  VarLabel::destroy(pRefinedLabel_preReloc);
  VarLabel::destroy(pSizeLabel);
  VarLabel::destroy(pSizeLabel_preReloc);
  VarLabel::destroy(pCurSizeLabel);
  VarLabel::destroy(pFiberDirLabel_preReloc);
  VarLabel::destroy(pFiberDirLabel);
  VarLabel::destroy(p_qLabel);
  VarLabel::destroy(p_qLabel_preReloc);
  VarLabel::destroy(pTemperatureGradientLabel);
  VarLabel::destroy(pTemperatureGradientLabel_preReloc);
  VarLabel::destroy(pPartitionUnityLabel);

  VarLabel::destroy(gAccelerationLabel);
  VarLabel::destroy(gColorLabel);
  VarLabel::destroy(gMatlsOnNodeLabel);
  VarLabel::destroy(gMassLabel);
  VarLabel::destroy(gMassAllLabel);
  VarLabel::destroy(gMassF0Label);
  VarLabel::destroy(gMassF1Label);
  VarLabel::destroy(gVelocityF0Label);
  VarLabel::destroy(gVelocityF1Label);
  VarLabel::destroy(gVelocityStarF0Label);
  VarLabel::destroy(gVelocityStarF1Label);
  VarLabel::destroy(gInternalForceF0Label);
  VarLabel::destroy(gInternalForceF1Label);
  VarLabel::destroy(gExternalForceF0Label);
  VarLabel::destroy(gExternalForceF1Label);
  VarLabel::destroy(gAccelerationF0Label);
  VarLabel::destroy(gAccelerationF1Label);
  VarLabel::destroy(gVelocityLabel);
  VarLabel::destroy(gVelocityBCLabel);
  VarLabel::destroy(gVelSPSSPLabel);
  VarLabel::destroy(gMatlProminenceLabel);
  VarLabel::destroy(gAlphaMaterialLabel);
  VarLabel::destroy(gNormAlphaToBetaLabel);
  VarLabel::destroy(gPositionLabel);
  VarLabel::destroy(gPositionF0Label);
  VarLabel::destroy(gPositionF1Label);
  VarLabel::destroy(gExternalForceLabel);
  VarLabel::destroy(gLSContactForceLabel);
  VarLabel::destroy(gInContactMatlLabel);
  VarLabel::destroy(gInternalForceLabel);
  VarLabel::destroy(gContactLabel);
  VarLabel::destroy(gVelocityStarLabel);
  VarLabel::destroy(gSurfaceAreaLabel);
  VarLabel::destroy(gSurfaceClayLabel);
  VarLabel::destroy(gSurfaceCementLabel);
  VarLabel::destroy(gNormTractionLabel);
  VarLabel::destroy(gNormTractionF0Label);
  VarLabel::destroy(gNormTractionF1Label);
  VarLabel::destroy(gStressLabel);
  VarLabel::destroy(gStressF0Label);
  VarLabel::destroy(gStressF1Label);
  VarLabel::destroy(gSurfNormLabel);
  VarLabel::destroy(gSurfNormF0Label);
  VarLabel::destroy(gSurfNormF1Label);
  VarLabel::destroy(gSurfLabel);
  VarLabel::destroy(gSurfGradLabel);
  VarLabel::destroy(gTemperatureLabel);
  VarLabel::destroy(gSp_volLabel); 
  VarLabel::destroy(gSp_vol_srcLabel); 
  VarLabel::destroy(gTemperatureNoBCLabel);
  VarLabel::destroy(gTemperatureStarLabel);
  VarLabel::destroy(gTemperatureRateLabel);
  VarLabel::destroy(gdTdtLabel);
  VarLabel::destroy(gHeatFluxLabel);
  VarLabel::destroy(gExternalHeatRateLabel);
  VarLabel::destroy(gExternalHeatFluxLabel);
  VarLabel::destroy(NC_CCweightLabel);
  VarLabel::destroy(gThermalContactTemperatureRateLabel);
  VarLabel::destroy(gStressForSavingLabel);
  VarLabel::destroy(gVolumeLabel);
  VarLabel::destroy(gVolumeF0Label);
  VarLabel::destroy(gVolumeF1Label);
  VarLabel::destroy(gZOILabel);
  VarLabel::destroy(MPMRefineCellLabel);
  VarLabel::destroy(cVolumeLabel);
  VarLabel::destroy(numLocInCellLabel);
  VarLabel::destroy(numInCellLabel);
  VarLabel::destroy(gradPAccNCLabel);
  VarLabel::destroy(dTdt_NCLabel);
  VarLabel::destroy(massBurnFractionLabel);
  VarLabel::destroy(dLdtDissolutionLabel);
  VarLabel::destroy(gCemVecLabel);
  VarLabel::destroy(NodalWeightSumLabel);
  VarLabel::destroy(frictionalWorkLabel);
  VarLabel::destroy(gNumNearParticlesLabel);

  VarLabel::destroy(partCountLabel);
  VarLabel::destroy(czCountLabel);
  VarLabel::destroy(lineSegmentCountLabel);
  VarLabel::destroy(TimeAveSpecificKELabel);
  VarLabel::destroy(doMechLabel);

  VarLabel::destroy(AccStrainEnergyLabel);
  VarLabel::destroy(StrainEnergyLabel);
  VarLabel::destroy(KineticEnergyLabel);
  VarLabel::destroy(AddedParticlesLabel);
  VarLabel::destroy(ThermalEnergyLabel);
  VarLabel::destroy(TotalMassLabel);
  VarLabel::destroy(InitialMassSVLabel);
  VarLabel::destroy(DissolvedMassLabel);
  VarLabel::destroy(PistonMassLabel);
  VarLabel::destroy(TotalSurfaceAreaLabel);
  VarLabel::destroy(TotalVolumeDeformedLabel);
  for(int iside=0;iside<6;iside++) {
      VarLabel::destroy(BndyContactAreaLabel[iside]);
      VarLabel::destroy(BndyContactCellAreaLabel[iside]);
      VarLabel::destroy(BndyForceLabel[iside]);
      VarLabel::destroy(BndyTractionLabel[iside]);
  }
  VarLabel::destroy(CenterOfMassPositionLabel);
  VarLabel::destroy(TotalMomentumLabel);
  VarLabel::destroy(RigidReactionForceLabel);
  VarLabel::destroy(TotalLocalizedParticleLabel);
  VarLabel::destroy(pCellNAPIDLabel);
  VarLabel::destroy(pCellNACZIDLabel);

  VarLabel::destroy(gVelocityOldLabel);
  VarLabel::destroy(dispNewLabel);
  VarLabel::destroy(dispIncLabel);
  VarLabel::destroy(pAccelerationLabel);
  VarLabel::destroy(dispIncQNorm0);
  VarLabel::destroy(dispIncNormMax);
  VarLabel::destroy(dispIncQNorm);
  VarLabel::destroy(dispIncNorm);
  VarLabel::destroy(pAccelerationLabel_preReloc);

  VarLabel::destroy(pDispLabel);
  VarLabel::destroy(pDispLabel_preReloc);
  VarLabel::destroy(gDisplacementLabel);

  // Destroy Material point erosion labels
  VarLabel::destroy(pErosionLabel);
  VarLabel::destroy(pErosionLabel_preReloc);

  // Destroy the MPM Physical BC pointer labels
  VarLabel::destroy(materialPointsPerLoadCurveLabel);
  VarLabel::destroy(pLoadCurveIDLabel);
  VarLabel::destroy(pLoadCurveIDLabel_preReloc);

   // Debugging labels
  VarLabel::destroy(pColorLabel);
  VarLabel::destroy(pColorLabel_preReloc);

  // For Cohesive Zones
  VarLabel::destroy(czAreaLabel);
  VarLabel::destroy(czAreaLabel_preReloc);
  VarLabel::destroy(czNormLabel);
  VarLabel::destroy(czNormLabel_preReloc);
  VarLabel::destroy(czTangLabel);
  VarLabel::destroy(czTangLabel_preReloc);
  VarLabel::destroy(czDispTopLabel);
  VarLabel::destroy(czDispTopLabel_preReloc);
  VarLabel::destroy(czDispBottomLabel);
  VarLabel::destroy(czDispBottomLabel_preReloc);
  VarLabel::destroy(czSeparationLabel);
  VarLabel::destroy(czSeparationLabel_preReloc);
  VarLabel::destroy(czForceLabel);
  VarLabel::destroy(czForceLabel_preReloc);
  VarLabel::destroy(czTopMatLabel);
  VarLabel::destroy(czTopMatLabel_preReloc);
  VarLabel::destroy(czBotMatLabel);
  VarLabel::destroy(czBotMatLabel_preReloc);
  VarLabel::destroy(czFailedLabel);
  VarLabel::destroy(czFailedLabel_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pYieldStressLabel);
  VarLabel::destroy(pYieldStressLabel_preReloc);
}
