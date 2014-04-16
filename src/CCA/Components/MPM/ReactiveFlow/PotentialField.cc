/*
 * PotentialField.cpp
 *
 *  Created on: Apr 12, 2014
 *      Author: hsrunner
 */

#include "PotentialField.h"

namespace Uintah {

PotentialField::PotentialField() {
	// TODO Auto-generated constructor stub

}

PotentialField::~PotentialField() {
	// TODO Auto-generated destructor stub
}
double PotentialField::gaoPotential(double conc, double concMax, double bolt,
						double detF, double omega, double mStress,
						double initPot){
	double potential = 0;

	potential = initPot + bolt * log(conc/(concMax - conc)) + omega * mStress;

	return potential;
}
} /* namespace UIntah */
