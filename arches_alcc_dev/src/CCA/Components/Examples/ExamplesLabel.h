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
 */


#ifndef Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h
#define Packages_Uintah_CCA_Components_Examples_ExamplesLabel_h

#include <vector>
using std::vector;


namespace Uintah {
  class VarLabel;
  class ExamplesLabel {
  public:

    // For SimpleCFD
    const VarLabel* bctype;
    const VarLabel* xvelocity;
    const VarLabel* yvelocity;
    const VarLabel* zvelocity;
    const VarLabel* density;
    const VarLabel* temperature;
    const VarLabel* pressure;
    const VarLabel* ccvelocity;

    const VarLabel* xvelocity_matrix;
    const VarLabel* xvelocity_rhs;
    const VarLabel* yvelocity_matrix;
    const VarLabel* yvelocity_rhs;
    const VarLabel* zvelocity_matrix;
    const VarLabel* zvelocity_rhs;
    const VarLabel* density_matrix;
    const VarLabel* density_rhs;
    const VarLabel* pressure_matrix;
    const VarLabel* pressure_rhs;
    const VarLabel* temperature_matrix;
    const VarLabel* temperature_rhs;

    const VarLabel* ccvorticity;
    const VarLabel* ccvorticitymag;
    const VarLabel* vcforce;
    const VarLabel* NN;

    // For AMRSimpleCFD
    const VarLabel* pressure2;
    const VarLabel* pressure2_matrix;
    const VarLabel* pressure2_rhs;

    const VarLabel* pressure_gradient_mag;
    const VarLabel* temperature_gradient_mag;
    const VarLabel* density_gradient_mag;


    // For ParticleTest1
    const VarLabel* pXLabel;
    const VarLabel* pXLabel_preReloc;
    const VarLabel* pMassLabel;
    const VarLabel* pMassLabel_preReloc;
    const VarLabel* pParticleIDLabel;
    const VarLabel* pParticleIDLabel_preReloc;

    vector<vector<const VarLabel*> > d_particleState;
    vector<vector<const VarLabel*> > d_particleState_preReloc;
    ExamplesLabel();
    ~ExamplesLabel();
  };
}

#endif
