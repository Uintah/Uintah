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

//----- PCProperties.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ColdFlow_h
#define Uintah_Component_Arches_ColdFlow_h

#include   <string>

/**
 * @class  PCProperties
 * @author Jeremy Thornock
 * @date   Nov 2018
 *
 * @brief PCProperties interface
 *
 * @todo
 *
 * @details
 *
 *
 *
*/


namespace Uintah {

class ArchesLabel;
class MPMArchesLabel;
class TimeIntegratorLabel;
class PCProperties : public MixingRxnModel {

public:

  PCProperties( MaterialManagerP& materialManager );

  ~PCProperties();

  void problemSetup( const ProblemSpecP& params );

  /** @brief Gets the thermochemical state for a patch **/
  void sched_getState( const LevelP& level,
                       SchedulerP& sched,
                       const int time_substep,
                       const bool initialize,
                       const bool modify_ref_den );

  /** @brief Gets the thermochemical state for a patch **/
  void getState( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 const int time_substep,
                 const bool initialize,
                 const bool modify_ref_den );

  typedef std::map<std::string, DepVarCont >       DepVarMap;
  typedef std::map<std::string, int >               IndexMap;

  void tableMatching(){};

  double getTableValue( std::vector<double>, std::string );

  double getTableValue( std::vector<double> iv, std::string depend_varname, StringToCCVar inert_mixture_fractions, IntVector c){ return -99;};

  double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions ){return -99;};

  int findIndex( std::string ){return 0; };

private:

}; // end class PCProperties

} // end namespace Uintah

#endif
