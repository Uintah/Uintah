/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * EwaldRealspace.cc
 *
 *  Created on: Jul 9, 2015
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Electrostatics/Ewald/Realspace.h>

using namespace Uintah;

void EwaldRealspace::generatePointScreeningMultipliers(const double& radius, // |R_ij|
                                              const double& beta,   // ewald screening parameter
                                                    double& B0,
                                                    double& B1,
                                                    double& B2,
                                                    double& B3)
{
  double beta_r = beta * radius;
  double inv_r = 1.0 / radius;
  double inv_r2 = inv_r * inv_r;
  double twoBeta2 = 2.0 * beta * beta;
  double invBetaRootPi = 1.0 / (beta * MDConstants::rootPI);
  double expNegBeta2r2 = exp(-beta_r * beta_r);

  B0 = erfc(beta_r)*inv_r;
  double rightSide = twoBeta2 * invBetaRootPi * expNegBeta2r2;
  B1 = (B0 + rightSide)*inv_r2;
  rightSide *= twoBeta2;
  B2 = (3.0 * B1 + rightSide)*inv_r2;
  rightSide *= twoBeta2;
  B3 = (5.0 * B2 + rightSide)*inv_r2;
}

void EwaldRealspace::generateTholeScreeningMultipliers(const double& a_Thole,
                                              const double& sqrt_alphai_alphaj,
                                              const double& radius,
                                                    double& B1,
                                                    double& B2,
                                                    double& B3)
{
  double A = a_Thole / sqrt_alphai_alphaj;
  double r2 = radius * radius;
  double invR2 = 1.0 / (r2);
  double denominator = invR2 / radius;
  double u = radius * r2 * A;

  double expTerm = exp(-u);
  double polyTerm = 1.0;
  B1 = (1.0 - polyTerm * expTerm) * denominator; // lambda_3
  denominator *= invR2;                          //    1 - exp(-au^3)

  polyTerm += u;
  B2 = (1.0 - polyTerm * expTerm) * denominator; // lambda_5
  denominator *= invR2;                          //    1 - (1 + au^3)exp(-au^3)
  polyTerm += 0.6 * u * u;
  B3 = (1.0 - polyTerm * expTerm) * denominator; // lambda_7
  return;                                        //    1 -
                                                 //     (1+au^3  (3/5)a^2u^6)*
                                                 //      exp(-au^3)

}

void EwaldRealspace::generateDipoleFunctionalTerms(const double&         q_i,
                                         const double&         q_j,
                                         const SCIRun::Vector& mu_i,
                                         const SCIRun::Vector& mu_j,
                                         const SCIRun::Vector& r_ij,
                                               double&         mu_jDOTr_ij,
                                               double&         G0,
                                               double&         G1_mu_q,
                                               double&         G1_mu_mu,
                                               double&         G2,
                                               SCIRun::Vector& gradG0,
                                               SCIRun::Vector& gradG1,
                                               SCIRun::Vector& gradG2)
{
  // GX and gradGX defined in:
  // Toukmaji et. al., J. Chem. Phys. 113(24), 10913-10927 (2000)
  // Eq. 2.10 and inline after Eq. 2.18
  double mu_iDOTmu_j = Dot(mu_i,mu_j);
  double mu_iDOTr_ij = Dot(mu_i,r_ij);
  mu_jDOTr_ij = Dot(mu_j,r_ij);  // Passed back for field calculation

  G0       =  q_i * q_j;
  G1_mu_q  =  q_j * mu_iDOTr_ij - q_i * mu_jDOTr_ij;
  G1_mu_mu =  mu_iDOTmu_j;
  G2       = -mu_iDOTr_ij * mu_jDOTr_ij;

  gradG0 =  MDConstants::V_ZERO;
  gradG1 =  q_i*mu_j - q_j * mu_i;
  gradG2 =  mu_i*(mu_jDOTr_ij) + mu_j*(mu_iDOTr_ij);

  return;
}

void EwaldRealspace::realspaceTholeDipole(
                                 const ProcessorGroup*      pg,
                                 const PatchSubset*         patches,
                                 const MaterialSubset*      atomTypes,
                                       DataWarehouse*       subOldDW,
                                       DataWarehouse*       subNewDW,
                                       DataWarehouse*       parentOldDW,
                                 const SimulationStateP*    simState,
                                 const MDLabel*             label,
                                 const double&              ewaldBeta,
                                 const double&              electrostaticCutoff,
                                 const int&                 electrostaticGhostCells
                                )
{
  size_t    numPatches      = patches->size();
  size_t    numAtomTypes    = atomTypes->size();
  double    cutoff2         = electrostaticCutoff * electrostaticCutoff;

  double    realElectrostaticEnergy = 0;
  Matrix3   realElectrostaticStress = MDConstants::M3_0;

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  /* Energy self term:       2
   *   -Beta    /  2   2 Beta       2  \
   *  -------  |  q  + --------|p_i|    |
   *  sqrt(PI)  \  i   3 sqrtPi        /
   *
   * Field self term:
   *        2
   *  4 Beta    ->
   *  --------  p
   *  3 sqrtPI   i
   */
  // Electrostatic constant (1/(4*PI*e_0)) in kCal*A/mol
  double chargeUnitNorm = 332.063712;

  double selfTermConstant = -ewaldBeta/sqrt(MDConstants::PI);
  double selfDipoleMult = 2.0*ewaldBeta*ewaldBeta/3.0;
  double selfFieldConst = -selfTermConstant*selfDipoleMult*2.0;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    // step through the materials for the reference sites
    for (size_t sourceTypeIndex = 0; sourceTypeIndex<numAtomTypes; ++sourceTypeIndex)
    {
      int             sourceType   = atomTypes->get(sourceTypeIndex);
      double          sourceCharge = (*simState)->getMDMaterial(sourceTypeIndex)
                                                  ->getCharge();
      double          sourcePol    = (*simState)->getMDMaterial(sourceTypeIndex)
                                                  ->getPolarizability();
      ParticleSubset* sourceSet    = parentOldDW->getParticleSubset(sourceType,
                                                                    currPatch);
      int sourceNum = sourceSet->numParticles();

      constParticleVariable<Point>  sourceX;
      parentOldDW->get(sourceX, label->global->pX, sourceSet);

      constParticleVariable<long64> sourceID;
      parentOldDW->get(sourceID, label->global->pID, sourceSet);

      constParticleVariable<Vector> sourceMu;
      subOldDW->get(sourceMu, label->electrostatic->pMu, sourceSet);

      ParticleVariable<SCIRun::Vector> localForce, localField;
      subNewDW->allocateAndPut(localForce,
                               label->electrostatic->pF_electroReal_preReloc,
                               sourceSet);
      subNewDW->allocateAndPut(localField,
                               label->electrostatic->pE_electroReal_preReloc,
                               sourceSet);

      // Initialize force, energy, and field
      // Energy and field get initialized with the self term subtractions
      for (int source = 0; source < sourceNum; ++ source) {
        localForce[source] = MDConstants::V_ZERO;
        localField[source] = selfFieldConst * sourceMu[source];
        realElectrostaticEnergy -= selfTermConstant*
              (sourceCharge*sourceCharge + sourceMu[source].length2());
      }

      for (size_t targetTypeIndex = 0; targetTypeIndex < numAtomTypes; ++targetTypeIndex)
      {
        size_t  targetType    = atomTypes->get(targetTypeIndex);
        double  targetCharge  = (*simState)->getMDMaterial(targetTypeIndex)
                                                   ->getCharge();
        double  targetPol     = (*simState)->getMDMaterial(targetTypeIndex)
                                                   ->getPolarizability();

        double  sqrt_alphai_alphaj  = sqrt(sourcePol*targetPol);
        ParticleSubset* targetSet;
        targetSet =  parentOldDW->getParticleSubset(targetType,
                                                    currPatch
//                                                    );
                                                    ,
                                                    Ghost::AroundCells,
                                                    electrostaticGhostCells,
                                                    label->global->pX);
        size_t targetNum = targetSet->numParticles();

        constParticleVariable<Point>  targetX;
        parentOldDW->get(targetX, label->global->pX, targetSet);

        constParticleVariable<long64> targetID;
        parentOldDW->get(targetID, label->global->pID, targetSet);

        constParticleVariable<Vector> targetMu;
        subOldDW->get(targetMu, label->electrostatic->pMu, targetSet);
        // loop over the local atoms
        for (int source=0; source < sourceNum; ++source) {
          SCIRun::Vector atomDipole = sourceMu[source];
          // loop over the neighbors
          for (size_t target=0; target < targetNum; ++target) {
            // Ensure i != j
            if (sourceID[source] != targetID[target])
            {
              offset = targetX[target] - sourceX[source];
//              coordSys->minimumImageDistance(sourceX[source],
//                                             targetX[target],
//                                             offset);
              double radius2 = offset.length2();
              // only calculate if neighbor within spherical cutoff around
              // local atom
              if (radius2 < cutoff2 ) {
                // double a_Thole = forcefield->getTholeScreeningParameter();
                double a_Thole = 0.2;
                SCIRun::Vector neighborDipole = targetMu[target];
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, ewaldBeta, B0, B1, B2, B3);
                double T1, T2, T3;
                generateTholeScreeningMultipliers(a_Thole,
                                                  sqrt_alphai_alphaj,
                                                  radius,
                                                  T1, T2, T3);
                double G0, G1_mu_q, G1_mu_mu, G2, mu_jDOTr_ij;
                SCIRun::Vector gradG0, gradG1, gradG2;
                generateDipoleFunctionalTerms(sourceCharge, targetCharge,
                                              atomDipole, neighborDipole,
                                              offset,
                                              mu_jDOTr_ij,
                                              G0, G1_mu_q, G1_mu_mu, G2,
                                              gradG0, gradG1, gradG2);
                // Dipole only terms:  G1_mu_mu, G2, gradG2
                // FIXME  The below setup is designed to apply Thole screening
                //        ONLY to pure Dipole terms.  This may not be the
                //        proper way to do it for forcefields other than
                //        Lucretius.  However, differences should be relatively
                //        small.
                realElectrostaticEnergy += ( B0*G0 + B1*G1_mu_q +
                                            (B1-T1)*G1_mu_mu + (B2-T2)*G2);
                SCIRun::Vector localForceVector = offset*( G0*B1
                                                        +  G1_mu_q*B2
                                                        +  G1_mu_mu * (B2 - T2)
                                                        +  G2 * (B3-T3) )
                                                          + ( B1 * gradG1
                                                           + (B2-T2) * gradG2 );
                localForce[source] += localForceVector;
                localField[source] += (sourceCharge*B1-mu_jDOTr_ij*B2)*offset
                                         + B1*neighborDipole;
                realElectrostaticStress += OuterProduct(offset,
                                                        localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
          localForce[source] *= chargeUnitNorm; // Insert dimensionalization constant here
          localField[source] *= 1.0;
          realElectrostaticStress *= chargeUnitNorm;
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  subNewDW->put(sum_vartype(0.5 * realElectrostaticEnergy),
              label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->put(matrix_sum(0.5 * realElectrostaticStress),
              label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

// Called from the SPME::calculate subscheduler as part of the polarizable
// iteration loop.
void EwaldRealspace::realspacePointDipole(
                                 const ProcessorGroup*      pg,
                                 const PatchSubset*         patches,
                                 const MaterialSubset*      atomTypes,
                                       DataWarehouse*       subOldDW,
                                       DataWarehouse*       subNewDW,
                                       DataWarehouse*       parentOldDW,
                                 const SimulationStateP*    simState,
                                 const MDLabel*             label,
                                 const double&              ewaldBeta,
                                 const double&              electrostaticCutoff,
                                 const int&                 electrostaticGhostCells
                                )
{
  size_t    numPatches     =   patches->size();
  size_t    numAtomTypes   =   atomTypes->size();
  double    cutoff2        =   electrostaticCutoff * electrostaticCutoff;

  double    realElectrostaticEnergy = 0;
  Matrix3   realElectrostaticStress = MDConstants::M3_0;
  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  /* Energy self term:       2
   *   -Beta    /  2   2 Beta       2  \
   *  -------  |  q  + --------|p_i|    |
   *  sqrt(PI)  \  i   3 sqrtPi        /
   *
   * Field self term:
   *        2
   *  4 Beta    ->
   *  --------  p
   *  3 sqrtPI   i
   */
  // Electrostatic constant (1/(4*PI*e_0)) in kCal*A/mol
  double chargeUnitNorm = 332.063712;

  double selfTermConstant = -ewaldBeta/sqrt(MDConstants::PI);
  double selfDipoleMult = 2.0*ewaldBeta*ewaldBeta/3.0;
  double selfFieldConst = -selfTermConstant*selfDipoleMult*2.0;
  double ewSelfEnergy = 0.0;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numAtomTypes; ++localIndex) {
      int       atomType    = atomTypes->get(localIndex);
      double    atomCharge  = (*simState)->getMDMaterial(localIndex)->getCharge();
      ParticleSubset* localSet = parentOldDW->getParticleSubset(atomType,
                                                                patch);
      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      constParticleVariable<Vector> localMu;
      parentOldDW->get(localX, label->global->pX, localSet);
      parentOldDW->get(localID, label->global->pID, localSet);
      subOldDW->get(localMu, label->electrostatic->pMu, localSet);

      size_t numLocalAtoms = localSet->numParticles();

      ParticleVariable<SCIRun::Vector> localForce, localField;
      subNewDW->allocateAndPut(localForce,
                               label->electrostatic->pF_electroReal_preReloc,
                               localSet);
      subNewDW->allocateAndPut(localField,
                               label->electrostatic->pE_electroReal_preReloc,
                               localSet);

      for (size_t localAtom = 0; localAtom < numLocalAtoms; ++localAtom)
      {
        localForce[localAtom]=MDConstants::V_ZERO;
        localField[localAtom]=MDConstants::V_ZERO;
        ewSelfEnergy += selfTermConstant * atomCharge * atomCharge;
      }

      for (size_t neighborIndex = 0;
                  neighborIndex < numAtomTypes;
                  ++neighborIndex)
      {
        int neighborType        = atomTypes->get(neighborIndex);
        double neighborCharge   = (*simState)->getMDMaterial(neighborIndex)
                                               ->getCharge();
        ParticleSubset* neighborSet;
        neighborSet =  parentOldDW->getParticleSubset(neighborType,
                                                      patch,
                                                      Ghost::AroundCells,
                                                      electrostaticGhostCells,
                                                      label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        constParticleVariable<Vector> neighborMu;
        parentOldDW->get(neighborX, label->global->pX, neighborSet);
        parentOldDW->get(neighborID, label->global->pID, neighborSet);
        subOldDW->get(neighborMu, label->electrostatic->pMu, neighborSet);

        size_t numNeighborAtoms = neighborSet->numParticles();

        // loop over the local atoms
        for (size_t localAtom=0; localAtom < numLocalAtoms; ++localAtom) {
          SCIRun::Vector atomDipole = localMu[localAtom];
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[localAtom] != neighborID[neighborAtom]) {
              offset = neighborX[neighborAtom] - localX[localAtom];
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                SCIRun::Vector neighborDipole = neighborMu[neighborAtom];
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, ewaldBeta, B0, B1, B2, B3);
                double G0, G1_mu_q, G1_mu_mu, G2, mu_jDOTr_ij;
                SCIRun::Vector gradG0, gradG1, gradG2;
                generateDipoleFunctionalTerms(atomCharge, neighborCharge,
                                              atomDipole, neighborDipole,
                                              offset,
                                              mu_jDOTr_ij,
                                              G0, G1_mu_q, G1_mu_mu, G2,
                                              gradG0, gradG1, gradG2);
                realElectrostaticEnergy         += (B0*G0 +
                                                    B1*(G1_mu_q + G1_mu_mu) +
                                                    B2*G2);
                SCIRun::Vector localForceVector  =-offset*(G0*B1 +
                                                          (G1_mu_q + G1_mu_mu)*B2
                                                         + G2*B3 )
                                                         + (B1*gradG1 + B2*gradG2);
                localForce[localAtom]   += localForceVector;
                localField[localAtom]   += (atomCharge*B1-mu_jDOTr_ij*B2)*offset
                                            + B1*neighborDipole;

                realElectrostaticStress += OuterProduct(offset, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
      for (size_t localAtom = 0; localAtom < numLocalAtoms; ++localAtom)
      {
        localForce[localAtom] *= chargeUnitNorm; //Insert dimensionalization constant here
        localField[localAtom] *= 1.0;
      }

    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  std::cout << "Self Term Energy: " << ewSelfEnergy << std::endl;
  subNewDW->put(sum_vartype((0.5 * realElectrostaticEnergy + ewSelfEnergy) * chargeUnitNorm),
              label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->put(matrix_sum(0.5 * realElectrostaticStress * chargeUnitNorm),
              label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

void EwaldRealspace::realspaceChargeOnly(
                                const ProcessorGroup*     pg,
                                const PatchSubset*        patches,
                                const MaterialSubset*     atomTypes,
                                      DataWarehouse*      subOldDW,
                                      DataWarehouse*      subNewDW,
                                      DataWarehouse*      parentOldDW,
                                const SimulationStateP*   simState,
                                const MDLabel*            label,
                                const double&             ewaldBeta,
                                const double&             electrostaticCutoff,
                                const int&                electrostaticGhostCells
                               )
{
  size_t    numPatches      = patches->size();
  size_t    numAtomTypes    = atomTypes->size();
  double    cutoff2         = electrostaticCutoff * electrostaticCutoff;

  double    realElectrostaticEnergy = 0;
  Matrix3   realElectrostaticStress = MDConstants::M3_0;

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* patch = patches->get(patchIndex);
    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numAtomTypes; ++localIndex)
    {

      int               localType   = atomTypes->get(localIndex);
      double            localCharge = (*simState)->getMDMaterial(localIndex)
                                                   ->getCharge();
      ParticleSubset*   localSet    = parentOldDW->getParticleSubset(localType,
                                                                     patch);
      size_t            numLocal    = localSet->numParticles();

      constParticleVariable<Point>  localX;
      parentOldDW->get(localX,  label->global->pX,  localSet);

      constParticleVariable<long64> localID;
      parentOldDW->get(localID, label->global->pID, localSet);

      ParticleVariable<SCIRun::Vector> localForce;
      subNewDW->allocateAndPut(localForce,
                               label->electrostatic->pF_electroReal_preReloc,
                               localSet);

      for (size_t neighborIndex = 0; neighborIndex < numAtomTypes; ++neighborIndex)
      {
        int     neighborType    = atomTypes->get(neighborIndex);
        double  neighborCharge  = (*simState)->getMDMaterial(neighborIndex)
                                               ->getCharge();
        ParticleSubset* neighborSet;
        neighborSet = parentOldDW->getParticleSubset(neighborType,
                                                     patch,
                                                     Ghost::AroundCells,
                                                     electrostaticGhostCells,
                                                     label->global->pX);
        size_t numNeighbor      = neighborSet->numParticles();

        constParticleVariable<Point>  neighborX;
        parentOldDW->get(neighborX,  label->global->pX,  neighborSet);

        constParticleVariable<long64> neighborID;
        parentOldDW->get(neighborID, label->global->pID, neighborSet);

        // loop over the local atoms
        for (size_t localAtom=0; localAtom < numLocal; ++localAtom)
        {
          localForce[localAtom]=MDConstants::V_ZERO;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighbor; ++neighborAtom)
          { // Ensure i != j
            if (localID[localAtom] != neighborID[neighborAtom])
            {
              offset = neighborX[neighborAtom] - localX[localAtom];
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around
              // local atom
              if (radius2 < cutoff2 ) {
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, ewaldBeta, B0, B1, B2, B3);
                double G0 = localCharge*neighborCharge;
                realElectrostaticEnergy += (B0*G0);
                SCIRun::Vector localForceVector = offset*(G0*B1);
                localForce[localAtom] += localForceVector;
                realElectrostaticStress += OuterProduct(offset,
                                                        localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
          localForce[localAtom] *= 1.0; // Insert dimensionalization constant here
          realElectrostaticStress *= 1.0;
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  subNewDW->put(sum_vartype(0.5 * realElectrostaticEnergy),
                label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->put(matrix_sum(0.5 * realElectrostaticStress),
                label->electrostatic->rElectrostaticRealStress);
  return;
} // End method
