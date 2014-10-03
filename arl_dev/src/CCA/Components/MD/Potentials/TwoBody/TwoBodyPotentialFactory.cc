/*
 * TwoBodyPotentialFactory.cc
 *
 *  Created on: Mar 12, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Forcefields/parseUtils.h>
#include <CCA/Components/MD/Forcefields/definedForcefields.h>
#include <CCA/Components/MD/Potentials/TwoBody/TwoBodyPotentialFactory.h>
#include <Core/Malloc/Allocator.h>
#include <CCA/Components/MD/Potentials/definedPotentials.h>
#include <cstdlib>

using namespace std;
using namespace Uintah;

NonbondedTwoBodyPotential* NonbondedTwoBodyFactory::create
                          (      Forcefield*                ff,
                           const std::string&               potentialType,
                           const std::vector<std::string>&  potentialArguments,
                           const std::string&               label,
                           const std::string&               comment)
{
    NonbondedTwoBodyPotential* nonbondedTwoBody = 0;

    std::string ffName = ff->getForcefieldDescriptor();
    if (ffName == "Lucretius") {
      if (potentialType == "exp-6") {
        double A = Parse::stringToDouble(potentialArguments[0])*ff->ffEnergyToInternal();
        double B = Parse::stringToDouble(potentialArguments[1])/ff->ffDistanceToInternal();
        double C = Parse::stringToDouble(potentialArguments[2])*ff->ffEnergyToInternal()*pow(ff->ffDistanceToInternal(),6);
        nonbondedTwoBody = scinew LucretiusExp6(A, B, C, label, comment);
      } // Lucretius::exp-6
      if (potentialType == "lj126") {
        double sigma, epsilon;
        double A = Parse::stringToDouble(potentialArguments[0])*ff->ffEnergyToInternal()*pow(ff->ffDistanceToInternal(),12);
        double C = Parse::stringToDouble(potentialArguments[2])*ff->ffEnergyToInternal()*pow(ff->ffDistanceToInternal(),6);
        sigma = pow(A/C, 1.0/6.0);
        epsilon = C/(4.0*pow(sigma, 6.0));
        nonbondedTwoBody = scinew LennardJonesPotential(sigma, epsilon, label, 12, 6, comment);
      } // Lucretius::lj126
      if (potentialType == "lj9-6") {
        double sigma, epsilon;
        double A = Parse::stringToDouble(potentialArguments[0])*ff->ffEnergyToInternal()*pow(ff->ffDistanceToInternal(),9);
        double C = Parse::stringToDouble(potentialArguments[2])*ff->ffEnergyToInternal()*pow(ff->ffDistanceToInternal(),6);
        sigma = pow(A/C, 1.0/3.0);
        epsilon = C/(4.0*pow(sigma, 6.0));
        nonbondedTwoBody = scinew LennardJonesPotential(sigma, epsilon, label, 9, 6, comment);
      } // Lucretius::lj9-6
    } // Lucretius

    else if (ffName == "Generic" ) { // Potential types defined for a generic, unspecified forcefield
      if (potentialType == "Buckingham" || potentialType == "Exp6" ) {
        double Rmin = Parse::stringToDouble(potentialArguments[0]);
        double Epsilon = Parse::stringToDouble(potentialArguments[1]);
        double Lambda = Parse::stringToDouble(potentialArguments[2]);
        nonbondedTwoBody = scinew Buckingham(Rmin, Epsilon, Lambda, label, comment);
      }
      if (potentialType == "LennardJones") {
        double sigma = Parse::stringToDouble(potentialArguments[0]);
        double epsilon = Parse::stringToDouble(potentialArguments[1]);
        int m = Parse::stringToInt(potentialArguments[2]);
        int n = Parse::stringToInt(potentialArguments[3]);
        nonbondedTwoBody = scinew LennardJonesPotential(sigma, epsilon, label, m, n, comment);
      }
    } // Generic

    return nonbondedTwoBody;

}


