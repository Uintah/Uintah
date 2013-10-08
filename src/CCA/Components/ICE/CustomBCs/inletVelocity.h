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

#ifndef ICE_INLETVELOCITY_h
#define ICE_INLETVELOCITY_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MersenneTwister.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Util/DebugStream.h>


static SCIRun::DebugStream coutBC_FC("ICE_BC_FC", false);
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct inletVel_variable_basket{
    int    verticalDir;       // which direction is vertical [0,1,2]
    
    // log law profile
    double roughness;              // aerodynamic roughness
    double vonKarman;              // vonKarman constant 
    
    // powerlaw profile
    double exponent;
    double maxHeight;              // max height of velocity profile before it's set to u_infinity
    
    Point gridMin;
    Point gridMax;
    
    // variance
    bool addVariance;             // add variance to the inlet velocity profile
    double C_mu;                   // constant
    double u_star;                // roughnes
    
  }; 
  //____________________________________________________________
  // This struct contains all of the additional local variables needed by setBC.
  struct inletVel_vars{
    constCCVariable<Vector> vel_CC;
    bool addVariance;
  };
  
  //____________________________________________________________
  bool read_inletVel_BC_inputs(const ProblemSpecP&,
                               inletVel_variable_basket* vb,
                               GridP& grid);
 
  void addRequires_inletVel(Task* t, 
                            const string& where,
                            ICELabel* lb,
                            const MaterialSubset* ice_matls,
                            const bool recursive);
                                                 
  void  preprocess_inletVelocity_BCs( DataWarehouse* old_dw,
                                      ICELabel* lb,
                                      const int indx,
                                      const Patch* patch,
                                      const string& where,
                                      bool& set_BCs,
                                      const bool recursive,
                                      inletVel_vars* inletVel_v);
                           
  int set_inletVelocity_BC(const Patch* patch,
                           const Patch::FaceType face,
                           CCVariable<Vector>& vel_CC,
                           const string& var_desc,
                           Iterator& bound_ptr,
                           const string& bc_kind,
                           const Vector& bc_value,
                           inletVel_variable_basket* inlet_var_basket,
                           inletVel_vars* inletVel_v );

/*______________________________________________________________________ 
 Purpose~   Sets the face center velocity boundary conditions
 ______________________________________________________________________*/
 template<class T>
 int set_inletVelocity_BCs_FC( const Patch* patch,
                               const Patch::FaceType face,
                               T& vel_FC,
                               Iterator& bound_ptr,
                               const string& bc_kind,
                               const double& bc_value,
                               inletVel_vars* inletVel_v,
                               inletVel_variable_basket* VB )
{

  coutBC_FC<< "Doing set_inletVelocity_BCs_FC: \t\t" 
            << "("<< bc_kind << ") \t\t" <<patch->getFaceName(face)<< endl;
  //__________________________________
  // on (x,y,z)minus faces move in one cell
  IntVector oneCell(0,0,0);
  if ( (face == Patch::xminus) || 
       (face == Patch::yminus) || 
       (face == Patch::zminus)){
   oneCell = patch->faceDirection(face);
  } 

  const Level* level = patch->getLevel();
  int vDir = VB->verticalDir;              // vertical direction
  int pDir = patch->getFaceAxes(face)[0];  // principal direction
  double d          = VB->gridMin(vDir);
  double gridHeight =  VB->gridMax(vDir);

  //__________________________________
  // 
  if( bc_kind == "powerLawProfile" ){
  
    double height     =  VB->maxHeight;      
    double U_infinity = bc_value;
    double n          = VB->exponent;
  
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr - oneCell;
      
       Point here   = level->getCellPosition(c);
       double h     = here.asVector()[vDir];
       double ratio = (h - d)/height;
       ratio = SCIRun::Clamp(ratio,0.0,1.0);  // clamp so 0< h/height < 1 in the edge cells 

       if( h > d && h < height){
         vel_FC[c] = U_infinity * pow(ratio, n);
       }else{                                // if height < h < gridHeight
         vel_FC[c] = U_infinity;
       }

       // Clamp edge/corner values 
       if( h < d || h > gridHeight ){
         vel_FC[c] = 0;
       }
  
     //std::cout << "        " << c <<  " h " << h  << " h/height  " << (h - d)/height << " vel_FC: " << vel_FC[c] <<endl;
    }
  }
  //__________________________________
  //
  else if( bc_kind == "logWindProfile" ){
  
    double inv_K       = 1.0/VB->vonKarman;
    double frictionVel = bc_value;
    double roughness   = VB->roughness;

    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr - oneCell;

      Point here   = level->getCellPosition(c);
      double z     = here.asVector()[vDir];
      double ratio = (z - d)/roughness;
      
      vel_FC[c]    = frictionVel * inv_K * log(ratio);

      // Clamp edge/corner values 
      if(z < d || z > gridHeight){
        vel_FC[c] = 0;
      }
      
//    std::cout << "        " << c <<  " z " << z  << " h/height  " << ratio << " vel_FC: " << vel_FC[c] <<endl;
    }
  }else{
    ostringstream warn;
    warn << "ERROR ICE::set_inletVelocity_BCs_FC  This type of boundary condition has not been implemented ("
         << bc_kind << ")\n" << endl; 
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  //______________________________________________________________________
  //  Addition of a 'kick' or variance to the mean velocity profile
  //  This matches the Turbulent Kinetic Energy profile of 1/sqrt(C_u) * u_star^2 ( 1- Z/height)^2
  if ( VB->addVariance) {
    
    constCCVariable<Vector> vel_CC = inletVel_v->vel_CC;
    
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr;
      IntVector cc = c - oneCell;
      
      Point here   = level->getCellPosition(c);
      double z     = here.asVector()[vDir];
      
      vel_FC[cc] = vel_CC[c][pDir];

      // Clamp edge/corner values 
      if(z < d || z > gridHeight){
        vel_FC[cc] = 0;
      }
      
      //if( cc.z() == 10){
      //  std::cout << "cc: " << cc << " c " << c << ", vel_CC.x, " << vel_CC[c].x()<< " vel_FC: " << vel_FC[cc] <<endl;
      //}
    }
  }  
  
  
  
  return bound_ptr.size(); 
}

} // End namespace Uintah
#endif
