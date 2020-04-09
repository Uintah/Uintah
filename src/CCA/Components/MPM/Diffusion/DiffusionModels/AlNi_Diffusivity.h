/*
 * AlNi_Diffusivity.h
 *
 *  Created on: Mar 24, 2017
 *      Author: jbhooper
 *
 *
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_ALNI_DIFFUSIVITY_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_ALNI_DIFFUSIVITY_H_

namespace Uintah {
  class AlNi
  {
    public:
      static inline double Diffusivity(const double T ) // Input is temperature
      {
          double R = 8.3144598; // Gas constant in J/(mol*K)
          double RTinv = 1.0/(8.3144598*T);
          double D = 0.0;
          if (T < 724)
          {
            D = 2.08e-7 * exp(-(92586*RTinv)); // Diffusivity m^2/s
          }
          else if (T > 860)
          {
            D = 1.81e-6 * exp(-(98646*RTinv)); // Diffusivity m^2/s
          }
          else
          {
            D = ((-1.9915e-18*T) + 4.4889e-15)*T-2.1627e-12;
          }
          return D;
      }
  };
}



#endif /* CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_ALNI_DIFFUSIVITY_H_ */
