/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>

#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Regridder.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Util/DebugStream.h>

using namespace Uintah;

FluidsBasedModel::FluidsBasedModel(const ProcessorGroup* myworld,
                                   const MaterialManagerP materialManager)
  : ModelInterface(myworld, materialManager)
{
}

FluidsBasedModel::~FluidsBasedModel()
{
  // delete transported Lagrangian variables
  std::vector<TransportedVariable*>::iterator t_iter;
  for(t_iter  = d_trans_vars.begin();
      t_iter != d_trans_vars.end(); t_iter++){
    TransportedVariable* tvar = *t_iter;
          
    VarLabel::destroy(tvar->var_Lagrangian);
    VarLabel::destroy(tvar->var_adv);
    delete tvar;
  }

  std::vector<AMRRefluxVariable*>::iterator r_iter;
  for( r_iter  = d_reflux_vars.begin();
       r_iter != d_reflux_vars.end(); r_iter++){
    AMRRefluxVariable* rvar = *r_iter;
          
    VarLabel::destroy(rvar->var_X_FC_flux);
    VarLabel::destroy(rvar->var_Y_FC_flux);
    VarLabel::destroy(rvar->var_Z_FC_flux);
    VarLabel::destroy(rvar->var_X_FC_corr);
    VarLabel::destroy(rvar->var_Y_FC_corr);
    VarLabel::destroy(rvar->var_Z_FC_corr);
  } 
}

void FluidsBasedModel::registerTransportedVariable(const MaterialSet* matlSet,
                                                   const VarLabel* var,
                                                   const VarLabel* src)
{
  TransportedVariable* t = scinew TransportedVariable;
  t->matlSet = matlSet;
  t->matls   = matlSet->getSubset(0);
  t->var = var;
  t->src = src;
  t->var_Lagrangian = VarLabel::create(var->getName()+"_L", var->typeDescription());
  t->var_adv        = VarLabel::create(var->getName()+"_adv", var->typeDescription());
  d_trans_vars.push_back(t);
}

//__________________________________
//  Register scalar flux variables needed
//  by the AMR refluxing task.  We're actually
//  creating the varLabels and putting them is a vector
void FluidsBasedModel::registerAMRRefluxVariable(const MaterialSet* matlSet,
                                            const VarLabel* var)
{
  AMRRefluxVariable* t = scinew AMRRefluxVariable;
  
  t->matlSet = matlSet;
  t->matls   = matlSet->getSubset(0);

  std::string var_adv_name = var->getName() + "_adv";

  t->var_adv = VarLabel::find(var_adv_name);  //Advected conserved quantity

  if( t->var_adv == nullptr ) {
    throw ProblemSetupException("\nError<>: The refluxing variable name("+var_adv_name +") could not be found",
                                   __FILE__, __LINE__);
  }
  
  t->var = var;
  
  t->var_X_FC_flux = VarLabel::create(var->getName()+"_X_FC_flux", 
                                SFCXVariable<double>::getTypeDescription());
  t->var_Y_FC_flux = VarLabel::create(var->getName()+"_Y_FC_flux", 
                                SFCYVariable<double>::getTypeDescription());
  t->var_Z_FC_flux = VarLabel::create(var->getName()+"_Z_FC_flux", 
                                SFCZVariable<double>::getTypeDescription());
                                
  t->var_X_FC_corr = VarLabel::create(var->getName()+"_X_FC_corr", 
                                SFCXVariable<double>::getTypeDescription());
  t->var_Y_FC_corr = VarLabel::create(var->getName()+"_Y_FC_corr", 
                                SFCYVariable<double>::getTypeDescription());
  t->var_Z_FC_corr = VarLabel::create(var->getName()+"_Z_FC_corr", 
                                SFCZVariable<double>::getTypeDescription());

  d_reflux_vars.push_back(t);
}
