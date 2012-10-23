/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *//*
 MMS.cc -  Supports three manufactured solutions

	   1) Axis Aligned MMS : Already was a part of Uintah. 
	   Paper : An evaluation of explicit time integration schemes for use with the generalized interpolation material point method ",
	   Volume 227, pp.9628Ã¢ÂÂ9642 2008
	   2) Generalized Vortex : Newly added
	   Paper : Establishing Credibility of Particle Methods through Verification testing. 
	   Particles 2011 II International Conference on Particle-based methods Fundamentals and Applications.
	   3) Expanding Ring : Newly added
	   Paper : An evaluation of explicit time integration schemes for use with the generalized interpolation material point method ",
	   Volume 227, pp.9628Ã¢ÂÂ9642 2008



Member Functions :

initializeParticleForMMS : Initilaizes the Particle data at t = 0 ; Some MMS have intial velocity/displacement/stress. 
			   For initial stress state, look at cnh_mms.cc 

computeExternalForceForMMS : Computes the analytically determined body force for the pre-determined deformation. 
			     Look at the papers mentioned above for more information.


		Author : Krishna Kamojjala
			 Department of Mechanical Engineering
			 University of Utah.
		Date   : 110824

*/


#ifndef __MMS_H__
#define __MMS_H__

#include <CCA/Components/MPM/SerialMPM.h>
#include <cmath>
#include <iostream>

namespace Uintah {

using namespace SCIRun;
using namespace std;

  class MMS {

	public :
	
	void initializeParticleForMMS(ParticleVariable<Point> &position,
				      ParticleVariable<Vector> &pvelocity,
                                      ParticleVariable<Matrix3> &psize,
                                      ParticleVariable<Vector> &pdisp,
                                      ParticleVariable<double> &pmass,
                                      ParticleVariable<double> &pvolume ,
                                      Point p, 
                                      Vector dxcc, 
                                      Matrix3 size , 
                                      const Patch* patch,
				      MPMFlags* flags,
				      particleIndex i );

        void computeExternalForceForMMS(DataWarehouse* old_dw,
					DataWarehouse* new_dw, 
					double time, 
					ParticleSubset* pset, 
					MPMLabel* lb, 
					MPMFlags* flags , 
					ParticleVariable<Vector> &ExtForce);

};

}// end namespace Uintah
#endif













