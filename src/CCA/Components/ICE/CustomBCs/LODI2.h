/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef Packages_Uintah_CCA_Components_Ice_LODI_h
#define Packages_Uintah_CCA_Components_Ice_LODI_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Core/Containers/StaticArray.h>

#include <sgi_stl_warnings_off.h>
#include <typeinfo>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  // press_infinity:  user input
  // sigma:           user input constant
  // iceMatl_indx:    user input, ice material index.
  
  struct Lodi_variable_basket{
    double press_infinity;  
    double sigma;
    int iceMatl_indx;
    vector<Patch::FaceType> LodiFaces;
    bool saveLiTerms;
  };    
  //____________________________________________________________
  // This struct contains the additional variables required to 
  // apply the Lodi Temperature, density and velocity BC.
  struct Lodi_vars{                
    Lodi_vars() : Li(6) {}  
    constCCVariable<double> speedSound;
    constCCVariable<double> gamma;   
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> press_CC;        
    constCCVariable<double> temp_CC;            
    SCIRun::StaticArray<CCVariable<Vector> > Li;        
  };
  
  void addRequires_Lodi(Task* t, 
                      const string& where,
                      ICELabel* lb,
                      const MaterialSubset* ice_matls,
                      Lodi_variable_basket* lv);
                      
  void preprocess_Lodi_BCs(DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          ICELabel* lb,
                          const Patch* patch,
                          const string& where,
                          const int indx,
                          SimulationStateP& sharedState,
                          bool& setLodiBcs,
                          Lodi_vars* lv,
                          Lodi_variable_basket* lvb);
                           

  bool read_LODI_BC_inputs(const ProblemSpecP&,
                           Lodi_variable_basket*);
                                               
  VarLabel* getMaxMach_face_VarLabel( Patch::FaceType face);                                           
                                                             
  void Lodi_maxMach_patchSubset(const LevelP& level,
                                 SimulationStateP& sharedState,
                                 vector<PatchSubset*> &);
                                  
  bool is_LODI_face(const Patch* patch,
                    Patch::FaceType face,
                    SimulationStateP& sharedState);                            
                            

  void computeLi(SCIRun::StaticArray<CCVariable<Vector> >& L,
                 const CCVariable<double>& rho,              
                 const CCVariable<double>& press,                   
                 const CCVariable<Vector>& vel,                  
                 const CCVariable<double>& speedSound,              
                 const Patch* patch,
                 DataWarehouse* new_dw,
                 SimulationStateP& sharedState,
                 const Lodi_variable_basket* user_inputs, 
                 const bool recursiveTasks);

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
                   SimulationStateP& sharedState);
                    
  void FaceTemp_LODI(const Patch* patch,
                    const Patch::FaceType face,
                    CCVariable<double>& temp_CC,
                    Lodi_vars* lv, 
                    const Vector& dx,
                    SimulationStateP& sharedState);
               
  void FacePress_LODI(const Patch* patch,
                      CCVariable<double>& press_CC,
                      SCIRun::StaticArray<CCVariable<double> >& rho_micro,
                      SimulationStateP& sharedState, 
                      Patch::FaceType face,
                      Lodi_vars* lv);

                          
} // End namespace Uintah
#endif
