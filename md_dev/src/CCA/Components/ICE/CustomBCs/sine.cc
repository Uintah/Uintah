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

#include <CCA/Components/ICE/CustomBCs/sine.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Math/MiscMath.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Math/MiscMath.h>
#include <typeinfo>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
namespace Uintah {
//__________________________________
//  To turn on couts
//  setenv SCI_DEBUG "SINE_DOING_COUT:+"
static DebugStream cout_doing("ICE_BC_CC", false);

/* ______________________________________________________________________
 Function~  read_Sine_BC_inputs--   
 Purpose~   -returns (true) if the sine BC is specified on any face,
            -reads input parameters thar are need by the setBC routines
 ______________________________________________________________________  */
bool read_Sine_BC_inputs(const ProblemSpecP& prob_spec,
                        sine_variable_basket* sine_vb)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if Slip/creep bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingSine = false;
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool is_a_Sine_face = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);

      if (bc_type["var"] == "Sine" && !is_a_Sine_face) {
        usingSine = true;
        is_a_Sine_face = true;
      }
    }
  }
  //__________________________________
  //  read in variables need by the boundary
  //  conditions and put them in the variable basket
  if(usingSine ){
    ProblemSpecP sine = bc_ps->findBlock("SINE_BC");
    if (!sine) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find SINE_BC block";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
    sine->require("omega", sine_vb->omega);
    sine->require("A",     sine_vb->A);
    sine->require("reference_pressure", sine_vb->p_ref);
    sine->require("reference_velocity",sine_vb->vel_ref);
    
    ProblemSpecP mat_ps= 
      prob_spec->findBlockWithOutAttribute("MaterialProperties");
    ProblemSpecP ice_ps= mat_ps->findBlock("ICE")->findBlock("material");

    ice_ps->require("gamma",        sine_vb->gamma);
    ice_ps->require("specific_heat",sine_vb->cv);
  }
  return usingSine;
}
/* ______________________________________________________________________ 
 Function~  addRequires_Sine--   
 Purpose~   requires 
 ______________________________________________________________________  */
void addRequires_Sine(Task* t, 
                      const string& where,
                      ICELabel* lb,
                      const MaterialSubset* /*ice_matls*/)
{
  cout_doing<< "Doing addRequires_Sine: \t\t" <<t->getName()
            << " " << where << endl;
  
  Ghost::GhostType  gn  = Ghost::None;
  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  
  if(where == "CC_Exchange"){
    t->requires(Task::NewDW, lb->press_CCLabel, press_matl,oims,gn, 0); 
   
  }
  if(where == "Advection"){
    t->requires(Task::NewDW, lb->press_CCLabel, press_matl,oims,gn, 0);    
  }
}
/*______________________________________________________________________ 
 Function~  preprocess_Sine_BCs-- 
 Purpose~   get data from the datawarehouse
______________________________________________________________________ */
void  preprocess_Sine_BCs(DataWarehouse* new_dw,
                          DataWarehouse* /*old_dw*/,
                          ICELabel* lb,
                          const int /*indx*/,
                          const Patch* patch,
                          const string& where,
                          bool& setSine_BCs,
                          sine_vars* sine_v)
{
  Ghost::GhostType  gn  = Ghost::None;
  setSine_BCs = false; 
  sine_v->where = where;
  //__________________________________
  //    Equilibrium pressure
  if(where == "EqPress"){
    setSine_BCs = true; 
    sine_v->delT = 0.0;  // Don't include delt at this point in the timestep
  }
  //__________________________________
  //    Explicit and semi-implicit update pressure
  if(where == "update_press_CC"){
    setSine_BCs = true; 
  }
  if(where == "implicitPressureSolve"){
    setSine_BCs=true;
  }
   
  if(where == "imp_update_press_CC"){
    setSine_BCs = true;
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setSine_BCs = true;
    new_dw->get(sine_v->press_CC, lb->press_CCLabel, 0, patch,gn,0);
    new_dw->get(sine_v->rho_CC,   lb->rho_CCLabel,   0, patch,gn,0);
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setSine_BCs = true;
    new_dw->get(sine_v->press_CC, lb->press_CCLabel, 0, patch,gn,0);
    new_dw->get(sine_v->rho_CC,   lb->rho_CCLabel,   0, patch,gn,0);
  }
}
/*_________________________________________________________________
 Function~ set_Sine_Velocity_BC--
 Purpose~  Set velocity boundary conditions
___________________________________________________________________*/
int  set_Sine_Velocity_BC(const Patch* patch,
                          const Patch::FaceType face,
                          CCVariable<Vector>& vel_CC,
                          const string& var_desc,
                          Iterator& bound_ptr,
                          const string& bc_kind,
                          SimulationStateP& sharedState,
                          sine_variable_basket* sine_var_basket,
                          sine_vars* sine_v)                     

{
  int nCells = 0;
  if (var_desc == "Velocity" && bc_kind == "Sine") {
    cout_doing << "    Vel_CC (Sine) \t\t" <<patch->getFaceName(face)<< endl;
    
    // bulletproofing
    if (!sine_var_basket || !sine_v){
      throw InternalError("set_Sine_velocity_BC", __FILE__, __LINE__);
    }
    
    double A       = sine_var_basket->A;
    double omega   = sine_var_basket->omega; 
    Vector vel_ref = sine_var_basket->vel_ref;           
    double t       = sharedState->getElapsedTime(); 
    t += sine_v->delT;
    double change  = A * sin(omega*t);
    
    Vector smallNum(1e-100);
    Vector one_or_zero = vel_ref/(vel_ref + smallNum);                                 
                                                      
    // only alter the velocity in the direction that the reference_velocity
    // is non-zero.       
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++)   {
      IntVector c = *bound_ptr;                                           
      vel_CC[c].x(vel_ref.x() +  one_or_zero.x() * change);
      vel_CC[c].y(vel_ref.y() +  one_or_zero.y() * change);  
      vel_CC[c].z(vel_ref.z() +  one_or_zero.z() * change);                                               
    }
    nCells += bound_ptr.size();
  }
  return nCells; 
}

/*_________________________________________________________________
 Function~ set_Sine_Temperature_BC--
 Purpose~  Set temperature boundary conditions
___________________________________________________________________*/
int set_Sine_Temperature_BC(const Patch* patch,
                            const Patch::FaceType face,
                            CCVariable<double>& temp_CC,
                            Iterator& bound_ptr,
                            const string& bc_kind,
                            sine_variable_basket* sine_var_basket,
                            sine_vars* sine_v)  
{
  int nCells = 0;
  if (bc_kind == "Sine") {
    cout_doing << "    Temp_CC (Sine) \t\t" <<patch->getFaceName(face)<< endl;

    // bulletproofing
    if (!sine_var_basket || !sine_v){
      throw InternalError("set_Sine_Temperature_BC", __FILE__, __LINE__);
    }
    double cv    = sine_var_basket->cv;
    double gamma = sine_var_basket->gamma;
    constCCVariable<double> press_CC = sine_v->press_CC;
    constCCVariable<double> rho_CC   = sine_v->rho_CC;
                                                                            
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {  
      IntVector c = *bound_ptr;                                             
      temp_CC[c]= press_CC[c]/((gamma - 1.0) * cv * rho_CC[c]);
    }
    nCells += bound_ptr.size();                                                   
  }
  return nCells;
} 

/*_________________________________________________________________
 Function~ set_Sine_press_BC--
 Purpose~  Set press boundary conditions
___________________________________________________________________*/
int set_Sine_press_BC(const Patch* patch,
                      const Patch::FaceType face,
                      CCVariable<double>& press_CC,
                      Iterator& bound_ptr,
                      const string& bc_kind,
                      SimulationStateP& sharedState,
                      sine_variable_basket* sine_var_basket,
                      sine_vars* sine_v)  
{
  cout_doing << "    press_CC (Sine) \t\t" <<patch->getFaceName(face)<< endl;

  // bulletproofing
  if (!sine_var_basket || !sine_v){
    throw InternalError("set_Sine_press_BC: sine_vars = null", __FILE__, __LINE__);
  }
  
  int nCells   = 0;      
  double A     =  sine_var_basket->A;
  double omega =  sine_var_basket->omega;   
  double p_ref =  sine_var_basket->p_ref;                               
  double t     =  sharedState->getElapsedTime();               
  t += sine_v->delT;  // delT is either 0 or delT 
  double change = A * sin(omega*t);                               

  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {  
    IntVector c = *bound_ptr;                                    
    press_CC[c] = p_ref + change;           
  }
  nCells += bound_ptr.size();
  return nCells;                                                  
}
  
}  // using namespace Uintah
