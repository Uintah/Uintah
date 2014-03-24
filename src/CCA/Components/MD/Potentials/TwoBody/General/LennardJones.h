/*
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
 */

/*
 * LennardJones.h
 *
 *  Created on: Jan 30, 2014
 *      Author: jbhooper
 */

#ifndef LENNARDJONES_H_
#define LENNARDJONES_H_

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

#include <string>

namespace Uintah {

	using namespace SCIRun;

	class LennardJonesPotential : public NonbondedTwoBodyPotential {
	  // Lennard-Jones potential  --            m                n  --
	  //                          |  /  sigma  \       /  sigma \     |
	  //   u(r) = 4.0 * epsilon * | |  -------  |  -  |  ------- |    |
	  //                          |	 \	 |r|   /       \   |r|  /     |
	  //		                  --                                --
	  //             r   du(r)
	  //   F(r) = - ---  -----
	  //            |r|	   dr
      public:
		//LennardJonesPotential() {}
		LennardJonesPotential(double,
		                      double,
                              const std::string&,
		                      size_t m = 6,
		                      size_t n = 6,
		                      const std::string& defaultComment = "");

		~LennardJonesPotential() {}

		void fillEnergyAndForce(SCIRun::Vector& force, double& energy, const SCIRun::Vector& offSet) const;
		inline void fillEnergyAndForce(SCIRun::Vector& force, double& energy, const SCIRun::Point& P1, const SCIRun::Point& P2) const {
		  fillEnergyAndForce(force, energy, P2 - P1);
		  return;
		}

		void fillEnergy(double& energy, const SCIRun::Vector& offSet) const;
		inline void fillEnergy(double& energy, const SCIRun::Point& P1, const SCIRun::Point& P2) const {
		  fillEnergy(energy, P2 - P1);
		  return;
		}

		void fillForce(SCIRun::Vector& force, const SCIRun::Vector& offSet) const;
		inline void fillForce(SCIRun::Vector& force, const SCIRun::Point& P1, const SCIRun::Point& P2) const {
		  fillForce(force, P2 - P1);
		  return;
		}

		const std::string getPotentialDescriptor() const {
		  return d_potentialDescriptor;
		}

		const std::string getComment() const {
		  return d_comment;
		}

		const std::string getLabel() const {
		  return d_label;
		}

  private:

		double RToPower(double r, double r2, int b) const;
		static const std::string d_potentialSubtype;
		static const int sc_maxIntegralPower;
		mutable std::string d_potentialDescriptor;
		const std::string d_comment;
		const std::string d_label;
		double sigma, epsilon;
		int m, n;
		double A, C;
	};

}



#endif /* LENNARDJONES_H_ */
