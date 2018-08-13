/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef Uintah_Component_Arches_TableLookup_h
#define Uintah_Component_Arches_TableLookup_h

// C++ includes
#include <string>

// Uintah includes
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <Core/Grid/MaterialManager.h>

namespace Uintah {

class WBCHelper;

class TableLookup {

public:

  enum TABLE_TYPE { CLASSIC, COLDFLOW, CONSTANT };

  TableLookup( MaterialManagerP& materialManager );

  ~TableLookup();

  void problemSetup( const ProblemSpecP& params );


  void sched_getState( const LevelP& level,
                       SchedulerP& sched,
                       const bool initialize,
                       const bool modify_ref_den,
                       const int time_substep );

  void sched_checkTableBCs( const LevelP& level,
                            SchedulerP& sched );

  void addLookupSpecies( );

  MixingRxnModel* get_table( std::string which_table="NA" ){
    if ( which_table == "NA" ){
      auto i = m_tables.begin();
      return i->second;
    }
    else {
      return m_tables[which_table];
    }
  }

  // HACK:
  // This is needed for the heat loss model in the current code
  TABLE_TYPE get_table_type(){ return m_table_type; }

  void set_bcHelper( WBCHelper* helper ){ m_bcHelper = helper; }

private:

  void sched_setDependBCs( const LevelP& level,
                           SchedulerP& sched,
                           MixingRxnModel* model );

  void setDependBCs( const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw, MixingRxnModel* model );

  std::map<std::string, MixingRxnModel*> m_tables;        ///< The lookup interface
  MaterialManagerP& m_materialManager;                    ///< Material manager
  TABLE_TYPE m_table_type;                                ///< Describes the table type
  WBCHelper* m_bcHelper;                                  ////< Interface to BCs

}; // end class ConstantProps

} // end namespace Uintah

#endif
