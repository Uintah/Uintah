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


#ifndef __DEFORMATION_STATE_DATA_H__
#define __DEFORMATION_STATE_DATA_H__

#include <Core/Math/Matrix3.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class DeformationState
    \brief A structure that stores deformation state data
    \author Biswajit Banerjee \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class DeformationState {

  public:
    double deltaT;                   // Delta t = t_n+1 - t_n
    Matrix3 velGrad;                 // l_{n+1} = grad(v)_{n+1}
    Matrix3 velGrad_old;             // l_n = grad(v)_{n}
    Matrix3 defGrad;                 // F_{n+1}
    Matrix3 defGrad_old;             // F_n
    Matrix3 defGrad_inc;             // Delta F = exp[Delta t * 0.5(l_n + l_{n+1})]
    Matrix3 rotation;                // R  in F = RU
    Matrix3 stretch;                 // U  in F = RU
    Matrix3 rateOfDef;               // d_{n+1} = 0.5*[l_{n+1} + l_{n+1}^T]
    Matrix3 spin;                    // w_{n+1} = 0.5*[l_{n+1} + l_{n+1}^T]
    double J;                        // J_{n+1} = det(F_{n+1})
    double J_inc;                    // Delta J = det(Delta F)
    Matrix3 strain;                  // eps = whatever strain measure is being used.
    double eps_v;                    // eps_v = trace(eps)
    Matrix3 dev_strain;              // eps_dev = eps - 1/3 eps_vol 1
    double eps_s;                    // eps_s = sqrt{2/3} ||eps_dev||

    DeformationState();

    DeformationState(const DeformationState& state);
    DeformationState(const DeformationState* state);

    ~DeformationState();

    DeformationState& operator=(const DeformationState& state);
    DeformationState* operator=(const DeformationState* state);

    void update(const Matrix3& velGrad_old, const Matrix3& velGrad, const Matrix3& defGrad,
                 const double& delT);
   
    void computeHypoelasticStrain();
    void computeGreenStrain();
    void computeAlmansiStrain();
    void computeCauchyGreenB();
    void computeCauchyGreenBbar();

  private:

    void copy(const DeformationState& state);
    void copy(const DeformationState* state);
    
  };

} // End namespace Uintah

#endif  // __DEFORMATION_STATE_DATA_H__ 
