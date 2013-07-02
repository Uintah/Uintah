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

#include <CCA/Components/ICE/CustomBCs/inletVelocity.h>
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
//  setenv SCI_DEBUG "INLETVEL_DOING_COUT:+"
static DebugStream cout_doing("ICE_BC_CC", false);

/* ______________________________________________________________________
 Purpose~   -returns (true) if the inletVel BC is specified on any face,
            -reads input parameters needed setBC routines
 ______________________________________________________________________  */
bool read_inletVel_BC_inputs(const ProblemSpecP& prob_spec,
                             inletVel_variable_basket* VB,
                             GridP& grid)
{
  //__________________________________
  // search the BoundaryConditions problem spec
  // determine if inletVelocity bcs are specified
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  bool usingBC = false;
  string whichProfile; 
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool setThisFace = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      whichProfile = bc_type["var"];
      if ( (whichProfile == "powerLawProfile" || whichProfile == "logProfile") && !setThisFace ) {
        usingBC = true;
        setThisFace = true;
        
        //__________________________________
        // bulletproofing
        if (bc_type["id"] == "all"){
          string warn="ERROR:\n Inputs:inletVelocity Boundary Conditions: You've specified the 'id' = all \n The 'id' must be the ice material.";
          throw ProblemSetupException(warn, __FILE__, __LINE__);  
        }
      }
    }
  }
  //__________________________________
  //  read in variables required by the boundary
  //  conditions and put them in the variable basket
  if(usingBC ){
  
    // set default values
   
    ProblemSpecP inlet_ps = bc_ps->findBlock("inletVelocity");
    if (!inlet_ps) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find inletVelocity_BC block";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
    inlet_ps -> get( "roughness",             VB->roughness   );
    inlet_ps -> get( "frictionVelocity",      VB->frictionVelocity );
    inlet_ps -> get( "exponent",              VB->exponent    );
    inlet_ps -> get( "U_infinity",            VB->U_infinity  );
    inlet_ps -> require( "verticalDirection", VB->verticalDir );
    
    // determine the orgin of the domain
    BBox b;
    grid->getInteriorSpatialRange(b);
    VB->gridOrigin = b.min();
    VB->gridHeight = b.max() - b.min();
  }
  return usingBC;
}
/* ______________________________________________________________________ 
 Purpose~   add the requires for the different task 
 ______________________________________________________________________  */
void addRequires_inletVel(Task* t, 
                          const string& where,
                          ICELabel* lb,
                          const MaterialSubset* /*ice_matls*/)
{
  cout_doing<< "Doing addRequires_inletVel: \t\t" <<t->getName()
            << " " << where << endl;
#if 0  
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
#endif
}

/*______________________________________________________________________ 
 Purpose~   get data from the datawarehouse
______________________________________________________________________ */
void  preprocess_inletVelocity_BCs(const string& where,
                                   bool& set_BCs)
{
  set_BCs = false; 
  //__________________________________
  //    Equilibrium pressure
  if(where == "EqPress"){
    set_BCs = false; 
  }
  //__________________________________
  //    Explicit and semi-implicit update pressure
  if(where == "update_press_CC"){
    set_BCs = false; 
  }
  if(where == "implicitPressureSolve"){
    set_BCs = false;
  }
   
  if(where == "imp_update_press_CC"){
    set_BCs = false;
  }
  //__________________________________
  //    cc_ Exchange
  if(where == "CC_Exchange"){
    set_BCs = true;
  }
  //__________________________________
  //    Advection
  if(where == "Advection"){
    set_BCs = true;
  }
}
/*_________________________________________________________________
 Purpose~  Set inlet velocity boundary conditions
___________________________________________________________________*/
int  set_inletVelocity_BC(const Patch* patch,
                          const Patch::FaceType face,
                          CCVariable<Vector>& vel_CC,
                          const string& var_desc,
                          Iterator& bound_ptr,
                          const string& bc_kind,
                          SimulationStateP& sharedState,
                          inletVel_variable_basket* VB )                     

{
  int nCells = 0;
  
  if (var_desc == "Velocity" && (bc_kind == "powerLawProfile" || bc_kind == "logProfile") ) {
    cout_doing << "    Vel_CC (" << bc_kind << ") \t\t" <<patch->getFaceName(face)<< endl;

    // bulletproofing
    if (!VB ){
      throw InternalError("set_inletVelocity_BC", __FILE__, __LINE__);
    }
    const Level* level = patch->getLevel();
    
    int nDir = patch->getFaceAxes(face)[0];  //normal velocity direction
    int vDir = VB->verticalDir;              // vertical direction
    double height = VB->gridHeight[vDir];
    
    // compute the velocity in the normal direction
    // u = U_infinity * (h/height)^n
    
    if( bc_kind == "powerLawProfile" ){
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++)   {
        IntVector c = *bound_ptr; 
        
        Point here = level->getCellPosition(c);
        double h = here.asVector()[vDir] - height;
        
        vel_CC[c].x();
        vel_CC[c].y();  
        vel_CC[c].z();                                               
      }
    }
    
    //   u = frictionVel * (1/vonKarman) * ln(h/roughness
    else if( bc_kind == "logLawProfile" ){
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++)   {
        IntVector c = *bound_ptr;                                           
        vel_CC[c].x();
        vel_CC[c].y();  
        vel_CC[c].z();                                               
      }
    }else{
    }
    nCells += bound_ptr.size();
  }

  return nCells; 
}
 
}  // using namespace Uintah
