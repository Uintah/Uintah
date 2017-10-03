/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>

#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/HeatFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/ArchesHeatFluxBC.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>

using namespace std;
using namespace Uintah;

std::vector<MPMPhysicalBC*> MPMPhysicalBCFactory::mpmPhysicalBCs;

void MPMPhysicalBCFactory::create(const ProblemSpecP& ps, const GridP& grid, const MPMFlags* flags)
{
  ProblemSpecP test = ps->findBlock("PhysicalBC");
  if (test){

    ProblemSpecP current_ps = ps->findBlock("PhysicalBC")->findBlock("MPM");


    for( ProblemSpecP child = current_ps->findBlock("force"); child != nullptr; child = child->findNextBlock("force") ) {
       mpmPhysicalBCs.push_back(scinew ForceBC(child));
    }

    for( ProblemSpecP child = current_ps->findBlock("pressure"); child != nullptr; child = child->findNextBlock("pressure") ) {
       mpmPhysicalBCs.push_back(scinew PressureBC(child, grid, flags));
    }

    for(ProblemSpecP child = current_ps->findBlock("scalar_flux"); child != nullptr; child = child->findNextBlock("scalar_flux") ) {
       mpmPhysicalBCs.push_back(scinew ScalarFluxBC(child, grid, flags));
    }

    for(ProblemSpecP child = current_ps->findBlock("heat_flux"); child != nullptr;  child = child->findNextBlock("heat_flux") ) {
       mpmPhysicalBCs.push_back(scinew HeatFluxBC(child, grid));
    }
    for(ProblemSpecP child = current_ps->findBlock("arches_heat_flux"); child != nullptr; child = child->findNextBlock("arches_heat_flux") ) {
       mpmPhysicalBCs.push_back( scinew ArchesHeatFluxBC( child, grid ) );
    }
  }
}

void
MPMPhysicalBCFactory::clean()
{
  for( int i = 0; i < static_cast<int>(mpmPhysicalBCs.size()); i++ ) {
    delete mpmPhysicalBCs[i];
  }
  mpmPhysicalBCs.clear();
}
