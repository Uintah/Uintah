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

#include <CCA/Components/ICE/ViscosityModel/Viscosity.h>
#include <Core/Grid/Patch.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Viscosity::Viscosity( ProblemSpecP& ps)
{
}

Viscosity::Viscosity( ProblemSpecP & boxps,
                      const GridP  & grid )
{
}

Viscosity::~Viscosity()
{
}

//______________________________________________________________________
//        Static methods
//______________________________________________________________________
//
//    The last model in the input file takes precedence over all models.
bool  Viscosity::isDynViscosityDefined( std::vector<bool> trueFalse )
{
  for( auto iter  = trueFalse.begin(); iter != trueFalse.end(); iter++){
    if( *iter ){
      return true;
    }
  }
  return false;
}

//______________________________________________________________________
//     If a model is out of order return false

bool Viscosity::inCorrectOrder( std::vector<Viscosity*> models )
{  
  int callOrder_old = 0;
  
  for( auto iter  = models.begin(); iter != models.end(); iter++){
    Viscosity* viscModel = *iter;
    
    int callOrder = viscModel->getCallOrder();
    if( callOrder < callOrder_old ){
      return false;
    }
    callOrder_old = callOrder;
  }
  
  return true;
}
