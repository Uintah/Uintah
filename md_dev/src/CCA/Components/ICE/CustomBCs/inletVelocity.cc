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
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/CellIterator.h>


static SCIRun::DebugStream cout_BC_CC("ICE_BC_CC", false);
namespace Uintah {
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
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face"); face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
    map<string,string> face;
    face_ps->getAttributes(face);
    bool setThisFace = false;
    
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      whichProfile = bc_type["var"];
      if ( (whichProfile == "powerLawProfile" || whichProfile == "logWindProfile") && !setThisFace ) {
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
    VB->vonKarman = 0.4;
   
    ProblemSpecP inlet_ps = bc_ps->findBlock("inletVelocity");
    if (!inlet_ps) {
      string warn="ERROR:\n Inputs:Boundary Conditions: Cannot find inletVelocity_BC block";
      throw ProblemSetupException(warn, __FILE__, __LINE__);
    }
    
    inlet_ps -> get( "roughness",             VB->roughness   );
    inlet_ps -> get( "vonKarmanConstant",     VB->vonKarman   );
    inlet_ps -> get( "exponent",              VB->exponent    );
    inlet_ps -> require( "verticalDirection", VB->verticalDir );
    
    // computational domain
    BBox b;
    grid->getInteriorSpatialRange(b);
    VB->gridMin = b.min();
    VB->gridMax = b.max();
  }
  return usingBC;
}

//______________________________________________________________________ 
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
                          const Vector& bc_value,
                          inletVel_variable_basket* VB )
{
  int nCells = 0;
  
  if (var_desc == "Velocity" && (bc_kind == "powerLawProfile" || bc_kind == "logWindProfile") ) {
    //cout_BC_CC << "    Vel_CC (" << bc_kind << ") \t\t" <<patch->getFaceName(face)<< endl;

    // bulletproofing
    if (!VB ){
      throw InternalError("set_inletVelocity_BC", __FILE__, __LINE__);
    }
    const Level* level = patch->getLevel();
    
    int nDir = patch->getFaceAxes(face)[0];  //normal velocity direction
    int vDir = VB->verticalDir;              // vertical direction
    

    //__________________________________
    // compute the velocity in the normal direction
    // u = U_infinity * pow( h/height )^n
    if( bc_kind == "powerLawProfile" ){
      double d          =  VB->gridMin(vDir);
      double height     =  VB->gridMax(vDir) - d;
      Vector U_infinity =  bc_value;
      double n          =  VB->exponent;
      
//      std::cout << "     height: " << height << " exponent: " << n << " U_infinity: " << U_infinity 
//           << " nDir: " << nDir << " vDir: " << vDir << endl;
           
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++)   {
        IntVector c = *bound_ptr; 
        
        Point here   = level->getCellPosition(c);
        double h     = here.asVector()[vDir] ;
        
        vel_CC[c]    = U_infinity;              // set the components that are not normal to the face
        double ratio = (h - d)/height;           
        
        ratio = SCIRun::Clamp(ratio,0.0,1.0);  // clamp so 0< h/height < 1 in the edge cells 
        
        vel_CC[c][nDir] = U_infinity[nDir] * pow(ratio, n);
        
        // Clamp edge/corner values 
        if( h < d || h > height ){
          vel_CC[c] = Vector(0,0,0);
        }
//        std::cout << "        " << c <<  " h " << h  << " h/height  " << ratio << " vel_CC: " << vel_CC[c] <<endl;                               
      }
    }
    
    //__________________________________
    //   u = U_star * (1/vonKarman) * ln( (z-d)/roughness)
    else if( bc_kind == "logWindProfile" ){
    
      double inv_K       = 1.0/VB->vonKarman;
      double d           = VB->gridMin(vDir);  // origin
      double gridMax     = VB->gridMax(vDir);
      Vector frictionVel = bc_value;
      double roughness   = VB->roughness;
      
//      std::cout << "     d: " << d << " frictionVel: " << frictionVel << " roughness: " << roughness 
//                << " nDir: " << nDir << " vDir: " << vDir << endl;
    
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++)   {
        IntVector c = *bound_ptr;
        
        Point here = level->getCellPosition(c);
        double z   = here.asVector()[vDir] ;
        
        vel_CC[c]    = frictionVel;            // set the components that are not normal to the face
        double ratio = (z - d)/roughness;
        
        vel_CC[c][nDir] = frictionVel[nDir] * inv_K * log(ratio);
        
        // Clamp edge/corner values 
        if(z < d || z > gridMax){
          vel_CC[c] = Vector(0,0,0);
        }

//        std::cout << "        " << c <<  " z " << z  << " z-d " << z-d << " ratio " << ratio << " vel_CC: " << vel_CC[c] <<endl;
      }
    }else{
      ostringstream warn;
      warn << "ERROR ICE::set_inletVelocity_BC  This type of boundary condition has not been implemented ("
           << bc_kind << ")\n" << endl; 
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    nCells += bound_ptr.size();
  }

  return nCells; 
}
 
}  // using namespace Uintah
