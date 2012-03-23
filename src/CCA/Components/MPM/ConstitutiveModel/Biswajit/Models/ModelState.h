/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __BB_PLASTICITY_STATE_DATA_H__
#define __BB_PLASTICITY_STATE_DATA_H__

#include <Core/Math/Matrix3.h>

namespace UintahBB {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ModelState
    \brief A structure that store the plasticity state data
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2003 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class ModelState {

  public:
    double yieldStress;
    double strainRate;
    double plasticStrainRate;
    double plasticStrain;
    double pressure;
    double temperature;
    double initialTemperature;
    double density;
    double initialDensity;
    double volume;
    double initialVolume;
    double bulkModulus;
    double initialBulkModulus;
    double shearModulus;
    double initialShearModulus;
    double meltingTemp;
    double initialMeltTemp;
    double specificHeat;
    double porosity;
    double energy;
    Matrix3 backStress;

    Matrix3 elasticStrain;
    Matrix3 elasticStrainTrial;

    double p;         // pressure = tr(sigma)
    double q;         // shear = sqrt(3J2); J2 = 1/2 s:s; s = sigma - p I
    double p_c;       // consolidation pressure

    double epse_v;    // volumetric elastic strain = tr(epse)
    double epse_s;    // deviatoric elastic strain = sqrt(2/3) ||ee||
                      //  ee = epse - 1/3 epse_v I
    double epse_v_tr; // trial volumetric elastic strain
    double epse_s_tr; // trial deviatoric elastic strain 

    ModelState();

    ModelState(const ModelState& state);
    ModelState(const ModelState* state);

    ~ModelState();

    ModelState& operator=(const ModelState& state);
    ModelState* operator=(const ModelState* state);
    
  };

} // End namespace Uintah

#endif  // __BB_PLASTICITY_STATE_DATA_H__ 
