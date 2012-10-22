/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/Regridder/RegridderFactory.h>
#include <CCA/Components/Regridder/HierarchicalRegridder.h>
#include <CCA/Components/Regridder/BNRRegridder.h>
#include <CCA/Components/Regridder/TiledRegridder.h>
#include <Core/Parallel/ProcessorGroup.h>

using namespace std;
using namespace Uintah;

RegridderCommon* RegridderFactory::create(ProblemSpecP& ps, 
                                          const ProcessorGroup* world)
{
  RegridderCommon* regrid = 0;
  
  ProblemSpecP amr = ps->findBlock("AMR");	
  ProblemSpecP reg_ps = amr->findBlock("Regridder");
  if (reg_ps) {

    string regridder;
    reg_ps->getAttribute( "type", regridder );

    if (world->myrank() == 0) {
      cout << "Using Regridder " << regridder << endl;
    }
    if(regridder == "Hierarchical") {
      regrid = scinew HierarchicalRegridder(world);
    } else if(regridder == "BNR") {
      regrid = scinew BNRRegridder(world);
    } else if (regridder == "Tiled") {
      regrid = scinew TiledRegridder(world);
    } else
      regrid = 0;
  }

  return regrid;

}
