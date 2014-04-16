/*
 * PotentialField.h
 *
 *  Created on: Apr 12, 2014
 *      Author: hsrunner
 */

#ifndef POTENTIALFIELD_H_
#define POTENTIALFIELD_H_

#include<cmath>
namespace Uintah {

class PotentialField {
public:
	PotentialField();
	virtual ~PotentialField();
	static double gaoPotential(double conc, double concMax, double bolt,
						double detF, double omega, double mStress,
						double initPot);
};

} /* namespace UIntah */

#endif /* POTENTIALFIELD_H_ */
