/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

//----- ColdFlow.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ColdFlow_h
#define Uintah_Component_Arches_ColdFlow_h

#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Util/DebugStream.h>

#include   <string>

/**
 * @class  ColdFlow
 * @author Jeremy Thornock
 * @date   Jan 2011
 *
 * @brief ColdFlow interface
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
class ColdFlow : public MixingRxnModel {

public:

  ColdFlow( ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~ColdFlow();

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

  /** @brief Compute cold flow density and temperature for simple two stream mixing */ 
  double coldFlowMixing( std::vector<double>& iv, int pos );

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

  double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions ){return -99;};

  int findIndex( std::string ){return 0; }; 

private:

  double d_stream[2][2];
  
  IntVector d_ijk_den_ref;                ///< Reference density location

  std::vector<std::string> d_allUserDepVarNames;    ///< Vector storing all independent varaible names requested in input file

  std::string d_cold_flow_mixfrac; 

  std::map<std::string, double> species_s1;
  std::map<std::string, double> species_s2;

}; // end class ColdFlow
  
} // end namespace Uintah

#endif
