#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# CMakeLists.txt for Packages/Uintah/CCA/Components/MPM

SET(Packages_Uintah_CCA_Components_MPM_SRCS
  SerialMPM.cc
  RigidMPM.cc
  MPMCommon.cc
  FractureMPM.cc
  ImpMPM.cc
  ShellMPM.cc
  AMRMPM.cc
  SimpleSolver.cc
  Solver.cc
  MPMBoundCond.cc
  MPMFlags.cc
  ImpMPMFlags.cc

  ConstitutiveModel/RigidMaterial.cc
  ConstitutiveModel/CompMooneyRivlin.cc
  ConstitutiveModel/ConstitutiveModelFactory.cc
  ConstitutiveModel/ConstitutiveModel.cc
  ConstitutiveModel/ImplicitCM.cc
  ConstitutiveModel/MPMMaterial.cc
  ConstitutiveModel/CompNeoHook.cc
  ConstitutiveModel/CNHDamage.cc
  ConstitutiveModel/CNHPDamage.cc
  ConstitutiveModel/CompNeoHookImplicit.cc
  ConstitutiveModel/TransIsoHyper.cc
  ConstitutiveModel/TransIsoHyperImplicit.cc
  ConstitutiveModel/ViscoTransIsoHyper.cc
  ConstitutiveModel/ViscoTransIsoHyperImplicit.cc
  ConstitutiveModel/CompNeoHookPlas.cc
  ConstitutiveModel/ViscoScram.cc
  ConstitutiveModel/ViscoSCRAMHotSpot.cc
  ConstitutiveModel/HypoElastic.cc
  ConstitutiveModel/HypoElasticImplicit.cc
  ConstitutiveModel/ViscoScramImplicit.cc
  ConstitutiveModel/MWViscoElastic.cc
  ConstitutiveModel/IdealGasMP.cc
  ConstitutiveModel/Membrane.cc 			
  ConstitutiveModel/ShellMaterial.cc 		
  ConstitutiveModel/HypoElasticPlastic.cc		
  ConstitutiveModel/ElasticPlastic.cc		
  ConstitutiveModel/Water.cc			
  ConstitutiveModel/SoilFoam.cc
  
  ConstitutiveModel/PlasticityModels/PlasticityState.cc
  ConstitutiveModel/PlasticityModels/DamageModel.cc
  ConstitutiveModel/PlasticityModels/DamageModelFactory.cc
  ConstitutiveModel/PlasticityModels/JohnsonCookDamage.cc
  ConstitutiveModel/PlasticityModels/HancockMacKenzieDamage.cc
  ConstitutiveModel/PlasticityModels/MPMEquationOfState.cc
  ConstitutiveModel/PlasticityModels/MPMEquationOfStateFactory.cc
  ConstitutiveModel/PlasticityModels/DefaultHypoElasticEOS.cc
  ConstitutiveModel/PlasticityModels/MieGruneisenEOS.cc
  ConstitutiveModel/PlasticityModels/PlasticityModel.cc
  ConstitutiveModel/PlasticityModels/PlasticityModelFactory.cc
  ConstitutiveModel/PlasticityModels/IsoHardeningPlastic.cc
  ConstitutiveModel/PlasticityModels/JohnsonCookPlastic.cc
  ConstitutiveModel/PlasticityModels/ZAPlastic.cc
  ConstitutiveModel/PlasticityModels/MTSPlastic.cc
  ConstitutiveModel/PlasticityModels/SCGPlastic.cc
  ConstitutiveModel/PlasticityModels/PTWPlastic.cc
  ConstitutiveModel/PlasticityModels/YieldCondition.cc
  ConstitutiveModel/PlasticityModels/YieldConditionFactory.cc
  ConstitutiveModel/PlasticityModels/GursonYield.cc
  ConstitutiveModel/PlasticityModels/VonMisesYield.cc
  ConstitutiveModel/PlasticityModels/StabilityCheck.cc
  ConstitutiveModel/PlasticityModels/StabilityCheckFactory.cc
  ConstitutiveModel/PlasticityModels/BeckerCheck.cc
  ConstitutiveModel/PlasticityModels/DruckerCheck.cc
  ConstitutiveModel/PlasticityModels/NoneCheck.cc
  ConstitutiveModel/PlasticityModels/DruckerBeckerCheck.cc
  ConstitutiveModel/PlasticityModels/ShearModulusModel.cc
  ConstitutiveModel/PlasticityModels/ShearModulusModelFactory.cc
  ConstitutiveModel/PlasticityModels/ConstantShear.cc
  ConstitutiveModel/PlasticityModels/MTSShear.cc
  ConstitutiveModel/PlasticityModels/NPShear.cc
  ConstitutiveModel/PlasticityModels/PTWShear.cc
  ConstitutiveModel/PlasticityModels/SCGShear.cc
  ConstitutiveModel/PlasticityModels/SpecificHeatModel.cc
  ConstitutiveModel/PlasticityModels/SpecificHeatModelFactory.cc
  ConstitutiveModel/PlasticityModels/ConstantCp.cc
  ConstitutiveModel/PlasticityModels/CopperCp.cc
  ConstitutiveModel/PlasticityModels/SteelCp.cc
  ConstitutiveModel/PlasticityModels/MeltingTempModel.cc
  ConstitutiveModel/PlasticityModels/MeltingTempModelFactory.cc
  ConstitutiveModel/PlasticityModels/ConstantMeltTemp.cc
  ConstitutiveModel/PlasticityModels/SCGMeltTemp.cc
  ConstitutiveModel/PlasticityModels/BPSMeltTemp.cc

  Contact/SpecifiedBodyContact.cc
  Contact/SingleVelContact.cc
  Contact/FrictionContact.cc
  Contact/ApproachContact.cc
  Contact/ContactFactory.cc
  Contact/CompositeContact.cc
  Contact/NullContact.cc
  Contact/ContactMaterialSpec.cc
  Contact/Contact.cc

  ThermalContact/ThermalContact.cc
  ThermalContact/STThermalContact.cc
  ThermalContact/NullThermalContact.cc
  ThermalContact/ThermalContactFactory.cc

  PhysicalBC/MPMPhysicalBCFactory.cc
  PhysicalBC/ForceBC.cc
  PhysicalBC/NormalForceBC.cc
  PhysicalBC/PressureBC.cc
  PhysicalBC/CrackBC.cc
  PhysicalBC/HeatFluxBC.cc
  PhysicalBC/ArchesHeatFluxBC.cc
  PhysicalBC/PolynomialData.cc		

  ParticleCreator/ParticleCreator.cc
  ParticleCreator/ParticleCreatorFactory.cc
  ParticleCreator/DefaultParticleCreator.cc
  ParticleCreator/ImplicitParticleCreator.cc
  ParticleCreator/FractureParticleCreator.cc
  ParticleCreator/MembraneParticleCreator.cc
  ParticleCreator/ShellParticleCreator.cc 

  Crack/ReadAndDiscretizeCracks.cc
  Crack/CrackGeometryFactory.cc
  Crack/CrackGeometry.cc
  Crack/NullCrack.cc
  Crack/QuadCrack.cc
  Crack/CurvedQuadCrack.cc
  Crack/TriangularCrack.cc
  Crack/ArcCrack.cc
  Crack/EllipticCrack.cc
  Crack/PartialEllipticCrack.cc
  Crack/ParticleNodePairVelocityField.cc
  Crack/CrackSurfaceContact.cc
  Crack/FractureParametersCalculation.cc
  Crack/CrackPropagation.cc
  Crack/MoveCracks.cc
  Crack/UpdateCrackFront.cc 

  HeatConduction/HeatConduction.cc
  HeatConduction/ImplicitHeatConduction.cc
)

IF (PETSC_LIBRARY)
  SET(Packages_Uintah_CCA_Components_MPM_SRCS ${Packages_Uintah_CCA_Components_MPM_SRCS} PetscSolver.cc)
ELSE (PETSC_LIBRARY)
  SET(Packages_Uintah_CCA_Components_MPM_SRCS ${Packages_Uintah_CCA_Components_MPM_SRCS} FakePetscSolver.cc)
ENDIF (PETSC_LIBRARY)


ADD_LIBRARY(Packages_Uintah_CCA_Components_MPM ${Packages_Uintah_CCA_Components_MPM_SRCS})

TARGET_LINK_LIBRARIES(Packages_Uintah_CCA_Components_MPM
  Packages_Uintah_CCA_Ports
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Util
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_Labels
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Parallel
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_GeometryPiece
  Packages_Uintah_Core_Math
  Core_Exceptions
  Core_Thread
  Core_Geometry
  Core_Util
  Core_Math
  ${XML2_LIBRARY}
  ${VT_LIBRARY}
  ${MPI_LIBRARY}
  ${M_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_CCA_Components_MPM)
ENDIF(BUILD_SHARED_LIBS)
