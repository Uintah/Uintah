/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/ICE/CustomBCs/MMS_BCs.h>
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
//  setenv SCI_DEBUG "MMS_DOING_COUT:+"
static DebugStream cout_doing("MMS_DOING_COUT", false);

/* ______________________________________________________________________
 Function~  read_MMS_BC_inputs--   
 Purpose~   -returns (true) if method of manufactured solutions BC is 
             specified on any face,
            -reads input parameters thar are need by the setBC routines
 ______________________________________________________________________  */
bool read_MMS_BC_inputs(const ProblemSpecP& prob_spec,
                        mms_variable_basket* mms_vb)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if Slip/creep bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingMMS = false;
  mms_vb->whichMMS = "none";
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool is_a_MMS_face = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);

      if (bc_type["var"] == "MMS_1" && !is_a_MMS_face) {
        usingMMS = true;
        is_a_MMS_face = true;
        mms_vb->whichMMS = bc_type["var"];
      }
    }
  }
  
  //__________________________________
  //  read in variables from ICE 
  if(usingMMS ){
    if(mms_vb->whichMMS == "MMS_1"){
      ProblemSpecP mat_ps= 
        prob_spec->findBlockWithOutAttribute("MaterialProperties");
      ProblemSpecP ice_ps= mat_ps->findBlock("ICE")->findBlock("material");

      ice_ps->require("gamma",mms_vb->gamma);
      ice_ps->require("dynamic_viscosity",  mms_vb->viscosity);
      ice_ps->require("specific_heat",mms_vb->cv);

      ProblemSpecP cfd_ice_ps = prob_spec->findBlock("CFD")->findBlock("ICE");
      ProblemSpecP c_init_ps  = cfd_ice_ps->findBlock("customInitialization");
      ProblemSpecP mms_ps     = c_init_ps->findBlock("manufacturedSolution");
      mms_ps->require("A", mms_vb->A);
    }
  }
  return usingMMS;
}
/* ______________________________________________________________________ 
 Function~  addRequires_MMS--   
 Purpose~   requires for all the tasks depends on which task you're in
 ______________________________________________________________________  */
void addRequires_MMS(Task* t, 
                     const string& where,
                     ICELabel* lb,
                     const MaterialSubset* /*ice_matls*/)
{
  cout_doing<< "Doing addRequires_MMS: \t\t" <<t->getName()
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
 Function~  preprocess_MMS_BCs-- 
 Purpose~   get data from dw and set where in the algorithm the
            MMS boundary condiitions are applied.
______________________________________________________________________ */
void  preprocess_MMS_BCs(DataWarehouse* new_dw,
                         DataWarehouse* /*old_dw*/,
                         ICELabel* lb,
                         const int /*indx*/,
                         const Patch* patch,
                         const string& where,
                         bool& setMMS_BCs,
                         mms_vars* mms_v)
{
  Ghost::GhostType  gn  = Ghost::None;
  setMMS_BCs = false; 
  mms_v->where = where;
  //__________________________________
  //    Equilibrium pressure
  if(where == "EqPress"){
    setMMS_BCs = true; 
  }
  //__________________________________
  //    update pressure
  if(where == "update_press_CC"){
    setMMS_BCs = true; 
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    setMMS_BCs = true; 
    new_dw->get(mms_v->press_CC, lb->press_CCLabel, 0, patch,gn,0);
    new_dw->get(mms_v->rho_CC,   lb->rho_CCLabel,   0, patch,gn,0);
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    setMMS_BCs = true; 
    new_dw->get(mms_v->press_CC, lb->press_CCLabel, 0, patch,gn,0);
    new_dw->get(mms_v->rho_CC,   lb->rho_CCLabel,   0, patch,gn,0); 
  }
}
/*_________________________________________________________________
 Function~ set_MMS_Velocity_BC--
 Purpose~  Set velocity boundary conditions using method of manufactured
           solution boundary conditions
___________________________________________________________________*/
int set_MMS_Velocity_BC(const Patch* patch,
                         const Patch::FaceType face,
                         CCVariable<Vector>& vel_CC,
                         const string& var_desc,
                         Iterator& bound_ptr,
                         const string& bc_kind,
                         SimulationStateP& sharedState,
                         mms_variable_basket* mms_var_basket,
                         mms_vars* mms_v)                     

{
  int nCells = 0;
  if (var_desc == "Velocity" && bc_kind == "MMS_1") {
    cout_doing << "Setting Vel_MMS on face " << face << endl;
    
    // bulletproofing
    if (!mms_var_basket || !mms_v){
      throw InternalError("set_MMS_velocity_BC", __FILE__, __LINE__);
    }
    if(bc_kind == "MMS_1") {
      double nu = mms_var_basket->viscosity;
      double A =  mms_var_basket->A;
      double t  = sharedState->getElapsedTime();
      t += mms_v->delT;
      
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        IntVector c = *bound_ptr;
        Point pt = patch->cellPosition(c);
        double x = pt.x(); 
        double y = pt.y();
        
        vel_CC[c].x( 1.0 - A * cos(x-t) * sin(y -t) * exp(-2.0*nu*t));
        vel_CC[c].y( 1.0 + A * sin(x-t) * cos(y -t) * exp(-2.0*nu*t));
        vel_CC[c].z(0.0);
      }
      nCells += bound_ptr.size();
    }
  } 
  return nCells;
}

/*_________________________________________________________________
 Function~ set_MMS_Temperature_BC--
 Purpose~  Set temperature boundary conditions using method of 
           manufactured solutions
___________________________________________________________________*/
int set_MMS_Temperature_BC(const Patch* /*patch*/,
                            const Patch::FaceType face,
                            CCVariable<double>& temp_CC,
                            Iterator& bound_ptr,
                            const string& bc_kind,
                            mms_variable_basket* mms_var_basket,
                            mms_vars* mms_v)  
{
  // bulletproofing
  if (!mms_var_basket || !mms_v){
    throw InternalError("set_MMS_Temperature_BC", __FILE__, __LINE__);
  }
  
  int nCells = 0;
  if (bc_kind == "MMS_1") {
    cout_doing << "Setting Temp_MMS on face " <<face<< endl;
    double cv = mms_var_basket->cv;
    double gamma = mms_var_basket->gamma;
    constCCVariable<double> press_CC = mms_v->press_CC;
    constCCVariable<double> rho_CC   = mms_v->rho_CC;

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr;
      temp_CC[c]= press_CC[c]/((gamma - 1.0) * cv * rho_CC[c]);
    }
    nCells += bound_ptr.size();
  }
  return nCells;
} 

/*_________________________________________________________________
 Function~ set_MMS_press_BC--
 Purpose~  Set press boundary conditions using method of 
           manufactured solutions
___________________________________________________________________*/
int set_MMS_press_BC(const Patch* patch,
                      const Patch::FaceType face,
                      CCVariable<double>& press_CC,
                      Iterator& bound_ptr,
                      const string& bc_kind,
                      SimulationStateP& sharedState,
                      mms_variable_basket* mms_var_basket,
                      mms_vars* mms_v)  
{
  cout_doing << "Setting press_MMS_BC on face " <<face<< endl;

  // bulletproofing
  if (!mms_var_basket || !mms_v){
    throw InternalError("set_MMS_press_BC: mms_vars = null", __FILE__, __LINE__);
  }
  
  int nCells = 0;
  if(bc_kind == "MMS_1") {
    double nu = mms_var_basket->viscosity;
    double A =  mms_var_basket->A;
    double t =  sharedState->getElapsedTime();
    double p_ref = 101325;
    t += mms_v->delT;

    for (bound_ptr.reset(); !bound_ptr.done();bound_ptr++) {
      IntVector c = *bound_ptr;
      Point pt = patch->cellPosition(c);
      double x = pt.x(); 
      double y = pt.y();
      press_CC[c] = p_ref - A*A/4.0 * exp(-4.0*nu*t)
                 *( cos(2.0*(x-t)) + cos(2.0*(y-t)) );
    }
    nCells += bound_ptr.size();
  }
  return nCells;
}
  
}  // using namespace Uintah
