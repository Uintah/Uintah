/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_MMS_BC_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_MMS_BC_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct mms_globalVars{
    double A;
    double viscosity;
    double gamma;
    double cv;
    std::string whichMMS;
  };    
  //____________________________________________________________
  // This struct contains all of the additional local variables needed by setBC.
  struct mms_localVars{
    constCCVariable<double> press_CC;
    constCCVariable<double> rho_CC;
    std::string where;
    double delT;
  };
  //____________________________________________________________
  
  
  bool read_MMS_BC_inputs(const ProblemSpecP&,
                          mms_globalVars* mms_vb);
                  
  void addRequires_MMS(Task* t, 
                       const std::string& where,
                       ICELabel* lb,
                       const MaterialSubset* ice_matls);
                       
  void  preprocess_MMS_BCs(DataWarehouse* new_dw,
                           DataWarehouse* old_dw,
                           ICELabel* lb,
                           const int indx,
                           const Patch* patch,
                           const std::string& where,
                           bool& setMMS_BCs,
                           mms_localVars* lv);
                           
  int  set_MMS_Velocity_BC(const Patch* patch,
                           const Patch::FaceType face,
                           CCVariable<Vector>& vel_CC,
                           const std::string& var_desc,
                           Iterator& bound_ptr,
                           const std::string& bc_kind,
                           SimulationStateP& sharedState,
                           mms_globalVars* gv,
                           mms_localVars* lv);
                           
  int  set_MMS_Temperature_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<double>& temp_CC,
                              Iterator& bound_ptr,
                              const std::string& bc_kind,
                              mms_globalVars* gv,
                              mms_localVars* lv);
                              
  int  set_MMS_press_BC(const Patch* patch,
                        const Patch::FaceType face,
                        CCVariable<double>& press_CC,
                        Iterator& bound_ptr,
                        const std::string& bc_kind,
                        SimulationStateP& sharedState,
                        mms_globalVars* gv,
                        mms_localVars* lv);  
                        
                        
/*______________________________________________________________________ 
 Function~  set_MMS_BCs_FC--
 Purpose~   Sets the face center velocity boundary conditions
 ______________________________________________________________________*/
 template<class T>
int set_MMS_BCs_FC( const Patch* patch,
                      const Patch::FaceType face,
                      T& vel_FC,
                      Iterator& bound_ptr,
                      const Vector& dx,
                      SimulationStateP& sharedState,
                      mms_globalVars* gv,
                      mms_localVars* lv)
{
  //cout<< "Doing set_MMS_BCs_FC: \t\t" << whichVel
  //          << " face " << face << endl;
 
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
  double nu = gv->viscosity;
  double A =  gv->A;
  double t  = sharedState->getElapsedSimTime();
  t += lv->delT;
    
  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
    IntVector c = *bound_ptr - oneCell;
    Point pt = patch->cellPosition(c);
    double x_CC = pt.x(); 
    double y_CC = pt.y();

    double x_FC = x_CC - (dx.x()/2) * x_one_zero;
    double y_FC = y_CC - (dx.y()/2) * y_one_zero;

/*`==========TESTING==========*/
#if 0
    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(6);
    if (c.y() ==25 && c.z() == 0 && (face == 0 || face == 1)){
      cout << "face " << face << " " << c 
            <<  " x_CC " << x_CC << " x_FC " << x_FC
            <<  " y_CC " << y_CC << " y_FC " << y_FC 
            << " t " << t << " nu " << nu << " A " << A <<endl;
    }

    if (c.x() ==25 && c.z() == 0 && (face == 2 || face == 3)){
      cout << "face " << face << " " << c 
            <<  " x_CC " << x_CC << " x_FC " << x_FC
            <<  " y_CC " << y_CC << " y_FC " << y_FC
            << " t " << t << " nu " << nu << " A " << A <<endl;
    }
#endif
/*===========TESTING==========`*/
    Vector vel(0.0,0.0,0.0);
    vel.x( 1.0 - A * cos(x_FC -t) * sin(y_FC -t) * exp(-2.0*nu*t));
    vel.y( 1.0 + A * sin(x_FC -t) * cos(y_FC -t) * exp(-2.0*nu*t));

    vel_FC[c] = x_one_zero * vel.x() 
              + y_one_zero * vel.y()
              + z_one_zero * vel.z();
  }
  int nCells = bound_ptr.size(); 
  return nCells; 
}                        
                                                
} // End namespace Uintah
#endif
