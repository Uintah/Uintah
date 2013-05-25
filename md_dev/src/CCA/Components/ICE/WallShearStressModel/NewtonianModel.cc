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

#include <CCA/Components/ICE/WallShearStressModel/NewtonianModel.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>

using namespace Uintah;
static DebugStream cout_doing("ICE_DOING_COUT", false);

NewtonianModel::NewtonianModel(ProblemSpecP& ps, SimulationStateP& sharedState)
  : WallShearStress(ps, sharedState)
{ 
}

NewtonianModel::~NewtonianModel()
{
}

//______________________________________________________________________
//  
void NewtonianModel::computeWallShearStresses(DataWarehouse* new_dw,
                                        const Patch* patch,
                                        const CCVariable<Vector>& vel_CC,
                                        const SFCXVariable<double>& uvel_FC,
                                        const SFCYVariable<double>& vvel_FC,
                                        const SFCZVariable<double>& wvel_FC,
                                        const CCVariable<double>& rho_CC,
                                        const int indx,
                                        SimulationStateP&  d_sharedState,
                                        CCVariable<double>& turb_viscosity)
{

}
