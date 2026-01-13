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

#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/ICE/WallShearStressModel/WallShearStress.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/MaterialManager.h>

using namespace Uintah;

WallShearStress::WallShearStress()
{
}

WallShearStress::WallShearStress( ProblemSpecP    & ps,
                                  MaterialManagerP& materialManager )
  : d_materialManager(materialManager)
{

  //__________________________________
  // bulletproofing  If using a wallShearStress model then all ice matls
  // must have a non-zero viscosity d
  unsigned int numMatls = d_materialManager->getNumMatls( "ICE" );

  for (unsigned int m = 0; m < numMatls; m++) {
    ICEMaterial* ice_matl = (ICEMaterial*) d_materialManager->getMaterial( "ICE", m);
    

    bool isViscosityDefined = ice_matl->isDynViscosityDefined();
    if( !isViscosityDefined ){
      std::string warn = "\nERROR:ICE:\n The viscosity can't be 0 when using a wallShearStress model";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
  }
}

WallShearStress::~WallShearStress()
{
}
