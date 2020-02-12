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

#ifndef ICE_INLETVELOCITY_h
#define ICE_INLETVELOCITY_h

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/ICE/Core/ICELabel.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>


static Uintah::DebugStream coutBC_FC("ICE_BC_FC", false);
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct inletVel_globalVars{
    int    verticalDir;           // which direction is vertical [0,1,2]
    int    iceMatl_indx;
    
    // log law profile
    double roughness;             // aerodynamic roughness
    double vonKarman;             // vonKarman constant 
    
    // powerlaw profile
    double exponent;
    double maxHeight;             // max height of velocity profile before it's set to u_infinity
    
    Point gridMin;
    Point gridMax;
    
    // variance
    bool addVariance;             // add variance to the inlet velocity profile
    double C_mu;                  // constant
    double u_star;                // roughnes
    
  }; 
  //____________________________________________________________
  // This struct contains additional local variables needed by setBC.
  struct inletVel_localVars{
    constCCVariable<Vector> vel_CC;
    bool addVariance;
  };
  
  //____________________________________________________________
  bool read_inletVel_BC_inputs(const ProblemSpecP&,
                               MaterialManagerP& materialManager,
                               inletVel_globalVars* global,
                               GridP& grid);
 
  void addRequires_inletVel(Task* t, 
                            const std::string& where,
                            ICELabel* lb,
                            const MaterialSubset* ice_matls,
                            const bool recursive);
                                                 
  void  preprocess_inletVelocity_BCs( DataWarehouse* old_dw,
                                      ICELabel* lb,
                                      const int indx,
                                      const Patch* patch,
                                      const std::string& where,
                                      bool& set_BCs,
                                      const bool recursive,
                                      inletVel_globalVars* global,
                                      inletVel_localVars* local );
                           
  int set_inletVelocity_BC(const Patch* patch,
                           const Patch::FaceType face,
                           CCVariable<Vector>& vel_CC,
                           const std::string& var_desc,
                           Iterator& bound_ptr,
                           const std::string& bc_kind,
                           const Vector& bc_value,
                           inletVel_globalVars* global,
                           inletVel_localVars* local );

/*______________________________________________________________________ 
 Purpose~   Sets the face center velocity boundary conditions
 ______________________________________________________________________*/
 template<class T>
 int set_inletVelocity_BCs_FC( const Patch* patch,
                               const Patch::FaceType face,
                               T& vel_FC,
                               Iterator& bound_ptr,
                               const std::string& bc_kind,
                               const double& bc_value,
                               inletVel_localVars* lv,
                               inletVel_globalVars* gv )
{

  coutBC_FC<< "Doing set_inletVelocity_BCs_FC: \t\t" 
            << "("<< bc_kind << ") \t\t" <<patch->getFaceName(face)<< std::endl;
  //__________________________________
  // on (x,y,z)minus faces move in one cell
  IntVector oneCell(0,0,0);
  if ( (face == Patch::xminus) || 
       (face == Patch::yminus) || 
       (face == Patch::zminus)){
   oneCell = patch->faceDirection(face);
  } 

  const Level* level = patch->getLevel();
  int vDir = gv->verticalDir;              // vertical direction
  int pDir = patch->getFaceAxes(face)[0];  // principal direction
  double d          = gv->gridMin(vDir);
  double gridHeight = gv->gridMax(vDir);

  //__________________________________
  // 
  if( bc_kind == "powerLawProfile" ){
  
    double height     =  gv->maxHeight;      
    double U_infinity = bc_value;
    double n          = gv->exponent;
  
    for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
      IntVector c = *bound_ptr - oneCell;
      
       Point here   = level->getCellPosition(c);
       double h     = here.asVector()[vDir];
       double ratio = (h - d)/height;
       ratio = Clamp(ratio,0.0,1.0);  // clamp so 0< h/height < 1 in the edge cells 

       if( h > d && h < height){
         vel_FC[c] = U_infinity * pow(ratio, n);
       }else{                                // if height < h < gridHeight
         vel_FC[c] = U_infinity;
       }

       // Clamp edge/corner values 
       if( h < d || h > gridHeight ){
         vel_FC[c] = 0;
       }
  
     //std::cout << "        " << c <<  " h " << h  << " h/height  " << (h - d)/height << " vel_FC: " << vel_FC[c] <<std::endl;
    }
  }
  //__________________________________
  //
  else if( bc_kind == "logWindProfile" ){
  
    double inv_K       = 1.0/gv->vonKarman;
    double frictionVel = bc_value;
    double roughness   = gv->roughness;

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
      
//    std::cout << "        " << c <<  " z " << z  << " h/height  " << ratio << " vel_FC: " << vel_FC[c] <<std::endl;
    }
  }else{
    std::ostringstream warn;
    warn << "ERROR ICE::set_inletVelocity_BCs_FC  This type of boundary condition has not been implemented ("
         << bc_kind << ")\n" << std::endl;
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }
  //______________________________________________________________________
  //  Addition of a 'kick' or variance to the mean velocity profile
  //  This matches the Turbulent Kinetic Energy profile of 1/sqrt(C_u) * u_star^2 ( 1- Z/height)^2
  if ( gv->addVariance) {
    
    constCCVariable<Vector> vel_CC = lv->vel_CC;
    
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
      //  std::cout << "cc: " << cc << " c " << c << ", vel_CC.x, " << vel_CC[c].x()<< " vel_FC: " << vel_FC[cc] <<std::endl;
      //}
    }
  }  
  
  
  
  return bound_ptr.size(); 
}

} // End namespace Uintah
#endif
