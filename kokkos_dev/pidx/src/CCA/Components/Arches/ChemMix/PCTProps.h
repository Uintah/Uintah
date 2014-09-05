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

//----- PCTProps.h --------------------------------------------------

#ifndef Uintah_Component_Arches_PCTProps_h
#define Uintah_Component_Arches_PCTProps_h

#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Util/DebugStream.h>

#include   <string>

/**
 * @class  Principle Component Transport Properties
 * @author Jeremy Thornock
 * @date   Feb 2013
 *
 * @brief Interface for getting properties from transported principle components
 *
 * @todo
 *
 * @details
 *
 *
*/


namespace Uintah {

class ArchesLabel; 
class MPMArchesLabel; 
class TimeIntegratorLabel; 
class BoundaryCondition_new; 
class MixingRxnModel;

class PCTProps : public MixingRxnModel {

  public: 

  PCTProps( ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~PCTProps();

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


  /** @brief A temporary solution to deal with boundary conditions on properties until Properties.cc is eliminated */ 
  void oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type );

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief Dummy initialization as required by MPMArches */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

	/** @brief Return a table lookup for a variable given the independent variable space. **/ 
  double getTableValue( std::vector<double>, std::string ); 

	/** @brief Match the requested dependent variable with their table index. **/ 
	void tableMatching(); 

	/** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - Grid wise **/
	double getTableValue( std::vector<double> iv, std::string depend_varname, StringToCCVar inert_mixture_fractions, IntVector c); 

	/** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - single point wise**/
	double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions );

  /** @brief Method to find the index for any dependent variable.  **/
  int inline findIndex( std::string name ){ return 0; };

  private: 

  BoundaryCondition_new* _boundary_condition; 


}; // end PCTProps
}  // end namespace Uintah

#endif
