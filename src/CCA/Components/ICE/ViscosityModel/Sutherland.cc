/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


#include <CCA/Components/ICE/ViscosityModel/Sutherland.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iostream>

using namespace Uintah;
//______________________________________________________________________
//    Cengel, Y., and Cimbala, J., "Fluid Mechanics Fundamentals and Applications, Third edition"
//    , McGraw-Hill Publishing, 2014, pg 53.
//______________________________________________________________________

Sutherland::Sutherland( ProblemSpecP& ps)
 : Viscosity(ps)
{
  ps->getWithDefault("a", d_a, 0.0);
  ps->getWithDefault("b", d_b, 0.0);

  set_isViscosityDefined( true ); // Assumption the viscosity will be non-zero everywhere
  setCallOrder( Middle );
  setName( "Sutherland" );
}

//______________________________________________________________________
//
Sutherland::~Sutherland()
{
}

//______________________________________________________________________
//
void Sutherland::outputProblemSpec(ProblemSpecP& vModels_ps)
{
  ProblemSpecP vModel = vModels_ps->appendChild("Model");
  vModel->setAttribute("name", "Sutherland");

  vModel->appendElement("a", d_a);
  vModel->appendElement("b", d_b);

}

//______________________________________________________________________
//
template< class CCVar>
void Sutherland::computeDynViscosity_impl( const Patch        * patch,
                                           CCVar              & temp_CC,
                                           CCVariable<double> & mu)
{
  CellIterator iter=patch->getCellIterator();
  for (;!iter.done();iter++) {
    IntVector c = *iter;

    double T = temp_CC[c];

    // Clamp minimum to 300 because of fitting forms
    if(T < 300.0){
      T = 300.0;
    }

    // Clamp maximum to 5000 because of fitting form
    if(T > 5000.0){
      T = 5000.0;
    }

    mu[c] = d_a * sqrt(T)/( 1.0 + (d_b/T) );
  }
}
