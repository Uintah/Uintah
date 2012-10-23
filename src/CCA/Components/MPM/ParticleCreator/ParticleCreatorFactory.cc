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

#include <CCA/Components/MPM/ParticleCreator/ParticleCreatorFactory.h>
#include <CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <CCA/Components/MPM/ParticleCreator/MembraneParticleCreator.h>
#include <CCA/Components/MPM/ParticleCreator/ShellParticleCreator.h>
#include <CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

ParticleCreator* ParticleCreatorFactory::create(ProblemSpecP& ps, 
                                                MPMMaterial* mat,
                                                MPMFlags* flags)
{

  ProblemSpecP cm_ps = ps->findBlock("constitutive_model");
  string mat_type;
  cm_ps->getAttribute("type",mat_type);

  if (flags->d_integrator_type == "implicit") 
    return scinew ImplicitParticleCreator(mat,flags);

  else if (flags->d_integrator_type == "fracture") 
    return scinew FractureParticleCreator(mat,flags);

  else if (mat_type == "membrane")
    return scinew MembraneParticleCreator(mat,flags);

  else if (mat_type == "shell_CNH")
    return scinew ShellParticleCreator(mat,flags);
  
  else
    return scinew ParticleCreator(mat,flags);

}


