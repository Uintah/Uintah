/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

// Arches includes
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/ChemMix/ColdFlow.h>
#include <CCA/Components/Arches/ChemMix/ConstantProps.h>

// Uintah includes
#include <Core/Exceptions/InvalidValue.h>

using namespace std;
namespace Uintah{

//--------------------------------------------------------------------------------------------------
TableLookup::TableLookup( SimulationStateP& sharedState ) : m_sharedState(sharedState)
{


}

//--------------------------------------------------------------------------------------------------
TableLookup::~TableLookup()
{
  for (auto i = m_tables.begin(); i != m_tables.end(); i++ ){
    delete i->second;
  }
}

//--------------------------------------------------------------------------------------------------
void
TableLookup::problemSetup( const ProblemSpecP& params )
{

  if ( !params->findBlock("Properties") ){

    proc0cout << "\n \n     WARNING: No tables (Chemistry of otherwise) found in the input file!\n \n " << std::endl;

  } else {

    ProblemSpecP db = params->findBlock("Properties");

    for ( ProblemSpecP db_tabs = db->findBlock("table"); db_tabs != nullptr; db_tabs = db_tabs->findNextBlock("table")){

      std::string type;
      std::string label;

      db_tabs->getAttribute("label", label);
      db_tabs->getAttribute("type", type);

      if ( type == "classic") {

        m_tables.insert(std::make_pair(label, scinew ClassicTableInterface( m_sharedState )));
        m_tables[label]->problemSetup(db_tabs);
        m_table_type = CLASSIC;

      } else if ( type == "coldflow") {

        m_tables.insert(std::make_pair(label, scinew ColdFlow( m_sharedState )));
        m_tables[label]->problemSetup(db_tabs);
        m_table_type = COLDFLOW;

      } else if ( type == "constant" ){

        m_tables.insert(std::make_pair(label, scinew ConstantProps( m_sharedState )));
        m_tables[label]->problemSetup(db_tabs);
        m_table_type = CONSTANT;

      } else {

        throw InvalidValue("ERROR!: No valid property model specified!",__FILE__,__LINE__);

      }

    }
  }

}

//--------------------------------------------------------------------------------------------------
void
TableLookup::sched_getState( const LevelP& level,
                             SchedulerP& sched,
                             const bool initialize,
                             const bool modify_ref_den,
                             const int time_substep )
{
  for ( auto i = m_tables.begin(); i != m_tables.end(); i++ ){

    i->second->sched_getState( level, sched, time_substep, initialize, modify_ref_den );

  }
}

//--------------------------------------------------------------------------------------------------
void
TableLookup::sched_checkTableBCs( const LevelP& level,
                                  SchedulerP& sched )
{
  for ( auto i = m_tables.begin(); i != m_tables.end(); i++ ){

    i->second->sched_checkTableBCs( level, sched );

  }
}

//--------------------------------------------------------------------------------------------------
void
TableLookup::addLookupSpecies( ){

  ChemHelper& helper = ChemHelper::self();
  std::vector<std::string>& sps = helper.model_req_species;
  std::vector<std::string>& old_sps = helper.model_req_old_species;

  std::vector<bool> sps_check(sps.size(), false);
  std::vector<bool> old_sps_check(old_sps.size(), false);

  for ( auto i_tab = m_tables.begin(); i_tab != m_tables.end(); i_tab++ ){

    int counter = 0;
    for ( auto i = sps.begin(); i != sps.end(); i ++ ){

      bool test = i_tab->second->insertIntoMap( *i );

      if ( test == true ){
        sps_check[counter] = true;
      }

      counter++;

    }

    counter = 0;

    for ( auto i = old_sps.begin(); i != old_sps.end(); i ++ ){

      bool test = i_tab->second->insertOldIntoMap( *i );

      if ( test == true ){
        old_sps_check[counter] = true;
      }

      counter++;

    }
  }

  //now make sure all species have been accounted for:
  for ( auto i = sps_check.begin(); i != sps_check.end(); i++ ){
    if ( !*i ){
      throw InvalidValue( "Error: The following species wasn\'t found in a table: "+*i,
                          __FILE__, __LINE__ );
    }
  }
  for ( auto i = old_sps_check.begin(); i != old_sps_check.end(); i++ ){
    if ( !*i ){
      throw InvalidValue( "Error: The following species wasn\'t found in a table: "+*i,
                          __FILE__, __LINE__ );
    }
  }
}


} //namespace Uintah
