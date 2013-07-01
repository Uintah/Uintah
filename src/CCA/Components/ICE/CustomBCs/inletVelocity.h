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
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct inletVel_variable_basket{
    double delT;
    double p_ref;
    Vector vel_ref;
  };    
  //____________________________________________________________
  // This struct contains all of the additional variables needed by setBC.
  struct inletVel_vars{
    string where;
    double delT;
  };
  //____________________________________________________________
  bool read_inletVel_BC_inputs(const ProblemSpecP&,
                               inletVel_variable_basket* vb);
                  
  void addRequires_inletVel(Task* t, 
                            const string& where,
                            ICELabel* lb,
                            const MaterialSubset* ice_matls);
                       
  void  preprocess_inletVel_BCs(DataWarehouse* new_dw,
                                DataWarehouse* old_dw,
                                ICELabel* lb,
                                const int indx,
                                const Patch* patch,
                                const string& where,
                                bool& setSine_BCs,
                                inletVel_vars* mss_v);
                           
  int set_inletVel_BC(const Patch* patch,
                      const Patch::FaceType face,
                      CCVariable<Vector>& vel_CC,
                      const string& var_desc,
                      Iterator& bound_ptr,
                      const string& bc_kind,
                      SimulationStateP& sharedState,
                      inletVel_variable_basket* inlet_var_basket,
                      inletVel_vars* inlet_v);
                        
                        
/*______________________________________________________________________ 
 Purpose~   Sets the face center velocity boundary conditions
 ______________________________________________________________________*/
 template<class T>
 int set_inletVel_BCs_FC( const Patch* patch,
                          const Patch::FaceType face,
                          T& vel_FC,
                          Iterator& bound_ptr,
                          SimulationStateP& sharedState,
                          inletVel_variable_basket* sine_var_basket,
                          inletVel_vars* sine_v)
{
//  cout<< "Doing set_sine_BCs_FC: \t\t" << whichVel   << " face " << face << endl;
  
  //__________________________________
  // on (x,y,z)minus faces move in one cell
  IntVector oneCell(0,0,0);
  if ( (face == Patch::xminus) || 
       (face == Patch::yminus) || 
       (face == Patch::zminus)){
    oneCell = patch->faceDirection(face);
  } 
  Vector one_or_zero = oneCell.asVector();
  
  //__________________________________
  //  set one or zero flags
  double x_one_zero = fabs(one_or_zero.x());
  double y_one_zero = fabs(one_or_zero.y());
  double z_one_zero = fabs(one_or_zero.z());

  //__________________________________
  //                            
  double A     = sine_var_basket->A;
  double omega = sine_var_basket->omega;
  Vector vel_ref=sine_var_basket->vel_ref;                                
  double t     = sharedState->getElapsedTime();                         
  t += sine_v->delT;     
  double change =   A * sin(omega*t);
                                             
  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {  
    IntVector c = *bound_ptr - oneCell;                  
      
    Vector vel(0.0,0.0,0.0);                                         
    vel.x(vel_ref.x() +  change); 
    vel.y(vel_ref.y() +  change );
    vel.z(vel_ref.z() +  change );  
                                                                     
    vel_FC[c] = x_one_zero * vel.x()                                 
              + y_one_zero * vel.y()                                 
              + z_one_zero * vel.z();                                
  } 
  return bound_ptr.size(); 
}                        
                                                
} // End namespace Uintah
#endif
