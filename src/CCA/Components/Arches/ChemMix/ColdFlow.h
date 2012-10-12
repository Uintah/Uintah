/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
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
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level, 
                       SchedulerP& sched, 
                       const TimeIntegratorLabel* time_labels, 
                       const bool initialize,
                       const bool with_energy_exch,
                       const bool modify_ref_den ); 

  /** @brief Gets the thermochemical state for a patch 
      @param initialize         Tells the method to allocateAndPut 
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw, 
                 const TimeIntegratorLabel* time_labels, 
                 const bool initialize, 
                 const bool with_energy_exch, 
                 const bool modify_ref_den );

  /** @brief Schedule computeHeatLoss -- Not used for this model. */
  void sched_computeHeatLoss( const LevelP& level, 
                              SchedulerP& sched, 
                              const bool intialize_me, const bool calcEnthalpy ){}; 

  /** @brief  schedules computeFirstEnthalpy -- Not used for this model. */
  void sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched ){}; 

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief Dummy initialization as required by MPMArches */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

        /** @brief Compute cold flow density and temperature for simple two stream mixing */ 
  double coldFlowMixing( std::vector<double>& iv, int pos );

  enum BoundaryType { DIRICHLET, NEUMANN, FROMFILE };

  struct DepVarCont {

    CCVariable<double>* var; 
    //int index; 

  }; 

  typedef std::map<string, DepVarCont >       DepVarMap;
  typedef std::map<string, int >               IndexMap; 

  /** @brief A temporary solution to deal with boundary conditions on properties until Properties.cc is eliminated */ 
  void oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type );

  double getTableValue( std::vector<double>, std::string ); 

	void tableMatching(){}; 

private:

  double d_stream[2][2];
  
  IntVector d_ijk_den_ref;                ///< Reference density location

  vector<string> d_allUserDepVarNames;    ///< Vector storing all independent varaible names requested in input file

  std::string d_cold_flow_mixfrac; 

  std::map<string, double> species_s1; 
  std::map<string, double> species_s2; 

}; // end class ColdFlow
  
} // end namespace Uintah

#endif
