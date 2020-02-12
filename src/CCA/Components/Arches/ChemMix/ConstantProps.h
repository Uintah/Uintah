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

//----- ConstantProps.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ConstantProps_h
#define Uintah_Component_Arches_ConstantProps_h

#include   <string>

/**
 * @class  ConstantProps
 * @author Jeremy Thornock
 * @date   Jan 2011
 *
 * @brief ConstantProps interface
 *
 * @todo
 *
 * @details
 * This class computes the density and temperature for two non-reacting
 * streams.

This code checks for the following tags/attributes in the input file:
The UPS interface is:

\code
<ColdFlow                       spec="OPTIONAL NO_DATA" >
  <mixture_fraction_label       spec="REQUIRED STRING"/>
  <Stream_1                     spec="MULTIPLE NO_DATA">
    <density                    spec="REQUIRED DOUBLE 'positive'"/>
    <temperature                spec="REQUIRED DOUBLE 'positive'"/>
  </Stream_1>
  <Stream_2                     spec="MULTIPLE NO_DATA">
    <density                    spec="REQUIRED DOUBLE 'positive'"/>
    <temperature                spec="REQUIRED DOUBLE 'positive'"/>
  </Stream_2>
</ColdFlow>
\endcode

 * If you have trouble reading your table, you can "setenv SCI_DEBUG TABLE_DEBUG:+" to get a
 * report of what is going on in the table reader.
 *
 *
*/


namespace Uintah {


class ArchesLabel;
class MPMArchesLabel;
class TimeIntegratorLabel;
class ConstantProps : public MixingRxnModel {

public:

  ConstantProps( MaterialManagerP& materialManager );

  ~ConstantProps();

  void problemSetup( const ProblemSpecP& params );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level,
                       SchedulerP& sched,
                       const int time_substep,
                       const bool initialize,
                       const bool modify_ref_den );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 const int time_substep,
                 const bool initialize,
                 const bool modify_ref_den );

  enum BoundaryType { DIRICHLET, NEUMANN, FROMFILE };

  struct DepVarCont {

    CCVariable<double>* var;
    //int index;

  };

  typedef std::map<std::string, DepVarCont >       DepVarMap;
  typedef std::map<std::string, int >               IndexMap;

  void tableMatching(){};

  double getTableValue( std::vector<double>, std::string );

  double getTableValue( std::vector<double> iv, std::string depend_varname, StringToCCVar inert_mixture_fractions, IntVector c){ return -99;};

  double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions, bool do_inverse = false ){return -99;};

  int findIndex( std::string ){return 0; };

private:

  IntVector d_ijk_den_ref;                ///< Reference density location

  double _density;
  double _temperature;
  bool _includeTemp;
}; // end class ConstantProps

} // end namespace Uintah

#endif
