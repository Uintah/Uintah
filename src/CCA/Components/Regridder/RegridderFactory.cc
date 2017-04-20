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

//-- Uintah component includes --//
#include <CCA/Components/Regridder/RegridderFactory.h>
#include <CCA/Components/Regridder/SingleLevelRegridder.h>
#include <CCA/Components/Regridder/TiledRegridder.h>

//-- Uintah framework includes --//
#include <Core/Parallel/ProcessorGroup.h>

using namespace Uintah;

RegridderCommon* RegridderFactory::create(ProblemSpecP& ps, const ProcessorGroup* world)
{
  RegridderCommon* regridder = 0;

  // Parse the AMR/Regridder portion of the input file
  ProblemSpecP amrPS = ps->findBlock("AMR");
  ProblemSpecP regridderPS = amrPS->findBlock("Regridder");

  if (regridderPS) {

    std::string regridderName;
    regridderPS->getAttribute("type", regridderName);

    if (world->myrank() == 0) {
      std::cout << "Using Regridder " << regridderName << std::endl;
    }
    if (regridderName == "Tiled") {
      regridder = scinew TiledRegridder(world);
    }
    else if (regridderName == "SingleLevel") {
      regridder = scinew SingleLevelRegridder(world);
    }
    else {
      regridder = 0;
    }
  }

  return regridder;

}
