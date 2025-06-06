/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef ICE_CUSTOMBCS_LODI_H
#define ICE_CUSTOMBCS_LODI_H

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/ICE/Core/ICELabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  // press_infinity:  user input
  // sigma:           user input constant
  struct Lodi_variable_basket{
    double press_infinity;  
    double sigma;
  };
    
  //____________________________________________________________
  // This struct contains the additional variables required to 
  // apply the Lodi Temperature, density and velocity BC.
  struct Lodi_vars{                
    Lodi_vars() : di(6) {}
    constCCVariable<double> rho_old;     
    constCCVariable<double> temp_old; 
    constCCVariable<double> speedSound;
    constCCVariable<double> cv;
    constCCVariable<double> gamma;   
    constCCVariable<Vector> vel_old;
    CCVariable<double> rho_CC;      // rho *after* BC has been applied
    CCVariable<Vector> vel_CC;      // vel *after* BC has been applied  
    CCVariable<double> press_tmp;        
    CCVariable<double> E;          // total energy
    CCVariable<Vector> nu;               
    std::vector<CCVariable<Vector> > di; 
    double delT;
    bool setLodiBcs; 
    Lodi_variable_basket* var_basket;           
  };
  //_____________________________________________________________
  // This struct contains the additional variables required to 
  // apply the Lodi pressure bcs.
  struct Lodi_vars_pressBC{
    Lodi_vars_pressBC(int numMatls): Temp_CC(numMatls), f_theta(numMatls), cv(numMatls),gamma(numMatls) {}
    std::vector<constCCVariable<double> > Temp_CC;
    std::vector<constCCVariable<double> > f_theta;
    std::vector<constCCVariable<double> > cv;
    std::vector<constCCVariable<double> > gamma;
    bool usingLODI;
    bool setLodiBcs;                  
  };

  void lodi_bc_preprocess( const Patch* patch,
                            Lodi_vars* lv,
                            ICELabel* lb,            
                            const int indx,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            MaterialManagerP& materialManager);
                            
  void lodi_getVars_pressBC( const Patch* patch,
                             Lodi_vars_pressBC* lodi_vars,
                             ICELabel* lb,
                             MaterialManagerP materialManager,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

  bool read_LODI_BC_inputs(const ProblemSpecP&,
                           Lodi_variable_basket*);
                                               
  VarLabel* getMaxMach_face_VarLabel( Patch::FaceType face);                                           
                                                             
  void Lodi_maxMach_patchSubset(const LevelP& level,
                                 MaterialManagerP& materialManager,
                                 vector<PatchSubset*> &);
                                  
  bool is_LODI_face(const Patch* patch,
                  Patch::FaceType face,
                  MaterialManagerP& materialManager);                            
                            
  void computeNu(CCVariable<Vector>& nu,
                 const CCVariable<double>& p, 
                 const Patch* patch,
                 MaterialManagerP& materialManager);  

  void computeDi(std::vector<CCVariable<Vector> >& d,
                 constCCVariable<double>& rho_old,  
                 const CCVariable<double>& press_tmp, 
                 constCCVariable<Vector>& vel_old, 
                 constCCVariable<double>& speedSound, 
                 const Patch* patch,
                 DataWarehouse*,
                 MaterialManagerP& materialManager);
                 
  double computeConvection(const double& nuFrt,     const double& nuMid, 
                           const double& nuLast,    const double& qFrt, 
                           const double& qMid,      const double& qLast,
                           const double& qConFrt,   const double& qConLast,
                           const double& deltaT,    const double& deltaX);

  void getBoundaryEdges(const Patch* patch,
                        const Patch::FaceType face,
                        vector<Patch::FaceType>& face0);
                                 
  int remainingVectorComponent(int dir1, int dir2);
  
  void FaceDensity_LODI(const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<double>& rho_CC,
                       Lodi_vars* lv,
                       const Vector& dx);
                  
  void FaceVel_LODI(const Patch* patch,
                   Patch::FaceType face,                 
                   CCVariable<Vector>& vel_CC,           
                   Lodi_vars* lv,
                   const Vector& dx,
                   MaterialManagerP& materialManager);
                    
  void FaceTemp_LODI(const Patch* patch,
                    const Patch::FaceType face,
                    CCVariable<double>& temp_CC,
                    Lodi_vars* lv, 
                    const Vector& dx,
                    MaterialManagerP& materialManager);
               
  void FacePress_LODI(const Patch* patch,
                      CCVariable<double>& press_CC,
                      std::vector<CCVariable<double> >& rho_micro,
                      MaterialManagerP& materialManager, 
                      Patch::FaceType face,
                      Lodi_vars_pressBC* lv);

                          
} // End namespace Uintah
#endif
