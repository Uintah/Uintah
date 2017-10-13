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

#ifndef Packages_Uintah_CCA_Components_Ice_Slip_h
#define Packages_Uintah_CCA_Components_Ice_Slip_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  struct slip_globalVars{
    double alpha_momentum;        // momentum accommidation coeff
    double alpha_temperature;     // temperature accomidation coeff.
    std::string SlipModel;        // which slip model  Deissler, Karniadakis-Beskok'
    bool        CreepFlow;        // include creep flow in the velocity calculation
  };


  //____________________________________________________________
  // This struct contains the additional variables required to compute
  // mean free path and the gradients
  struct slip_localVars{
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> temp_CC;
    constCCVariable<double> press_CC;
    constCCVariable<double> gamma;
    constCCVariable<double> specific_heat;
    constCCVariable<double> thermalCond;
    constCCVariable<double> viscosity;
    CCVariable<double> lamda;             //  mean free path
    double alpha_momentum;                //  momentum accomodation coefficient
    double alpha_temperature;             //  thermal accomodation coefficient
    std::string SlipModel;                //  slip model
    bool   CreepFlow;                     //  include creep flow, or not
  };

  bool read_MicroSlip_BC_inputs(const ProblemSpecP&,
                                slip_globalVars* gv);

  void addRequires_MicroSlip(Task                 * t,
                             const std::string    & where,
                             ICELabel             * lb,
                             const MaterialSubset * ice_matls,
                             slip_globalVars      * sv);


  void preprocess_MicroSlip_BCs(DataWarehouse     * old_dw,
                                DataWarehouse     * new_dw,
                                ICELabel          * lb,
                                const Patch       * patch,
                                const std::string & where,
                                const int           indx,
                                SimulationStateP  & sharedState,
                                bool              & setSlipBcs,
                                slip_localVars    * lv,
                                slip_globalVars   * gv);

  bool is_MicroSlip_face(const Patch      * patch,
                         Patch::FaceType    face,
                         SimulationStateP & sharedState);

  int set_MicroSlipVelocity_BC(const Patch          * patch,
                              const Patch::FaceType   face,
                              CCVariable<Vector>    & vel_CC,
                              const std::string     & var_desc,
                              Iterator              & bound_ptr,
                              const std::string     & bc_kind,
                              const Vector            wall_velocity,
                              slip_localVars        * lv);

  int  set_MicroSlipTemperature_BC(const Patch          * patch,
                                   const Patch::FaceType   face,
                                   CCVariable<double>    & temp_CC,
                                   Iterator              & bound_ptr,
                                   const std::string     & bc_kind,
                                   const double            wall_temperature,
                                   slip_localVars        * lv);
} // End namespace Uintah
#endif
