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

#include "InternalVar_BorjaPressure.h"
#include <cmath>
#include <iostream>
#include <Core/Exceptions/InvalidValue.h>

using namespace UintahBB;
using namespace Uintah;
using namespace std;


InternalVar_BorjaPressure::InternalVar_BorjaPressure(ProblemSpecP& ps)
{
  ps->require("pc0",d_pc0);
  ps->require("lambdatilde",d_lambdatilde);
  ps->require("kappatilde",d_kappatilde);

  // Initialize internal variable labels for evolution
  pPcLabel = VarLabel::create("p.p_c",
        ParticleVariable<double>::getTypeDescription());
  pPcLabel_preReloc = VarLabel::create("p.p_c+",
        ParticleVariable<double>::getTypeDescription());
}
         
InternalVar_BorjaPressure::InternalVar_BorjaPressure(const InternalVar_BorjaPressure* cm)
{
  d_pc0 = cm->d_pc0;
  d_lambdatilde = cm->d_lambdatilde;
  d_kappatilde = cm->d_kappatilde;

  // Initialize internal variable labels for evolution
  pPcLabel = VarLabel::create("p.p_c",
        ParticleVariable<double>::getTypeDescription());
  pPcLabel_preReloc = VarLabel::create("p.p_c+",
        ParticleVariable<double>::getTypeDescription());
}
         
InternalVar_BorjaPressure::~InternalVar_BorjaPressure()
{
  VarLabel::destroy(pPcLabel);
  VarLabel::destroy(pPcLabel_preReloc);
}


void InternalVar_BorjaPressure::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP int_var_ps = ps->appendChild("internal_variable_model");
  int_var_ps->setAttribute("type","borja_consolidation_pressure");

  int_var_ps->appendElement("pc0",d_pc0);
  int_var_ps->appendElement("lambdatilde",d_lambdatilde);
  int_var_ps->appendElement("kappatilde",d_kappatilde);
}

         
void 
InternalVar_BorjaPressure::addInitialComputesAndRequires(Task* task,
                                                   const MPMMaterial* matl ,
                                                   const PatchSet*)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pPcLabel, matlset);
}

void 
InternalVar_BorjaPressure::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl ,
                                   const PatchSet*)
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pPcLabel, matlset,Ghost::None);
  task->computes(pPcLabel_preReloc, matlset);
}

void 
InternalVar_BorjaPressure::addParticleState(std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to)
{
  from.push_back(pPcLabel);
  to.push_back(pPcLabel_preReloc);
}

void 
InternalVar_BorjaPressure::allocateCMDataAddRequires(Task* task,
                                               const MPMMaterial* matl ,
                                               const PatchSet* ,
                                               MPMLabel* )
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pPcLabel_preReloc, matlset, Ghost::None);
}

void 
InternalVar_BorjaPressure::allocateCMDataAdd(DataWarehouse* old_dw,
                                       ParticleSubset* addset,
                                       map<const VarLabel*, 
                                         ParticleVariableBase*>* newState,
                                       ParticleSubset* delset,
                                       DataWarehouse* new_dw )
{
  ParticleVariable<double> pPc;
  constParticleVariable<double> o_Pc;

  new_dw->allocateTemporary(pPc,addset);

  new_dw->get(o_Pc,pPcLabel_preReloc,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o = delset->begin(); o != delset->end(); o++, n++) {
    pPc[*n] = o_Pc[*o];
  }

  (*newState)[pPcLabel]=pPc.clone();

}

void 
InternalVar_BorjaPressure::initializeInternalVariable(ParticleSubset* pset,
                                                      DataWarehouse* new_dw)
{
  ParticleVariable<double> pPc;
  new_dw->allocateAndPut(pPc, pPcLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
    pPc[*iter] = d_pc0;
  }
}

void 
InternalVar_BorjaPressure::getInternalVariable(ParticleSubset* pset ,
                                               DataWarehouse* old_dw,
                                               constParticleVariableBase& pPc) 
{
  old_dw->get(pPc, pPcLabel, pset);
}

void 
InternalVar_BorjaPressure::allocateAndPutInternalVariable(ParticleSubset* pset,
                                                          DataWarehouse* new_dw,
                                                          ParticleVariableBase& pPc_new) 
{
  new_dw->allocateAndPut(pPc_new, pPcLabel_preReloc, pset);
}

void
InternalVar_BorjaPressure::allocateAndPutRigid(ParticleSubset* pset ,
                                               DataWarehouse* new_dw,
                                               constParticleVariableBase& pPc)
{
  ParticleVariable<double> pPc_new;
  new_dw->allocateAndPut(pPc_new, pPcLabel_preReloc, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
     pPc_new[*iter] = dynamic_cast<constParticleVariable<double>& >(pPc)[*iter];
  }
}

////////////////////////////////////////////////////////////////////////////////////////
//  Compute the internal variable
double 
InternalVar_BorjaPressure::computeInternalVariable(const ModelState* state) const
{
  // Get old p_c
  double pc_n = state->p_c;  // Old Pc

  // Get the trial elastic strain and the updated elastic strain
  // (volumetric part)
  double strain_elast_v_tr = state->epse_v_tr;
  double strain_elast_v = state->epse_v;

  // Calculate new p_c
  double pc = pc_n*exp(-(strain_elast_v_tr-strain_elast_v)/(d_lambdatilde-d_kappatilde));
  return pc;
}

////////////////////////////////////////////////////////////////////////////////////////
// Compute derivative of internal variable with respect to volumetric
// elastic strain
double 
InternalVar_BorjaPressure::computeVolStrainDerivOfInternalVariable(const ModelState* state) const
{
  // Get old p_c
  double pc_n = state->p_c;

  // Get the trial elastic strain and the updated elastic strain
  // (volumetric part)
  double strain_elast_v_tr = state->epse_v_tr;
  double strain_elast_v = state->epse_v;

  // Calculate  dp_c/depse_v
  double pc = pc_n*exp(-(strain_elast_v_tr-strain_elast_v)/(d_lambdatilde-d_kappatilde));
  return pc/(d_lambdatilde-d_kappatilde);;
}
