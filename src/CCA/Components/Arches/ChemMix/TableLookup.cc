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

// Arches includes
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/Arches/ChemMix/ClassicTableInterface.h>
#include <CCA/Components/Arches/ChemMix/ColdFlow.h>
#include <CCA/Components/Arches/ChemMix/ConstantProps.h>
#include <CCA/Components/Arches/WBCHelper.h>
#include <CCA/Components/Arches/ArchesMaterial.h>

// Uintah includes
#include <Core/Exceptions/InvalidValue.h>

using namespace std;
namespace Uintah{

//--------------------------------------------------------------------------------------------------
TableLookup::TableLookup( MaterialManagerP& materialManager ) : m_materialManager(materialManager)
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

        m_tables.insert(std::make_pair(label, scinew ClassicTableInterface( m_materialManager )));
        m_tables[label]->problemSetup(db_tabs);
        m_table_type = CLASSIC;

      } else if ( type == "coldflow") {

        m_tables.insert(std::make_pair(label, scinew ColdFlow( m_materialManager )));
        m_tables[label]->problemSetup(db_tabs);
        m_table_type = COLDFLOW;

      } else if ( type == "constant" ){

        m_tables.insert(std::make_pair(label, scinew ConstantProps( m_materialManager )));
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

    sched_setDependBCs( level, sched, i->second );

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
TableLookup::sched_setDependBCs( const LevelP& level,
                                 SchedulerP& sched,
                                 MixingRxnModel* model ){

  Task* tsk = scinew Task("setDependBCs", this, &TableLookup::setDependBCs, model );

  std::map<std::string, const VarLabel*> depend_var_map = model->getDVVars();

  for ( auto i = depend_var_map.begin(); i != depend_var_map.end(); i++ ){
    tsk->modifies( i->second );
  }

  sched->addTask( tsk, level->eachPatch(), m_materialManager->allMaterials( "Arches" ) );

}

//--------------------------------------------------------------------------------------------------
void
TableLookup::setDependBCs( const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           MixingRxnModel* model )
{
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = m_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    const BndMapT& bc_info = m_bcHelper->get_boundary_information();

    for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

      if (i_bc->second.has_patch(patch->getID()) ){

        std::string facename = i_bc->second.name;

        std::map<std::string, const VarLabel*> depend_var_map = model->getDVVars();
        for ( auto i_var = depend_var_map.begin(); i_var != depend_var_map.end(); i_var++ ){

          CCVariable<double> var;
          new_dw->getModifiable( var, i_var->second, indx, patch );

          std::string varname = i_var->first;

          const BndCondSpec* spec = i_bc->second.find(varname);

          if ( spec != NULL ){

            Uintah::ListOfCellsIterator& cell_iter =
              m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

            if ( (*spec).bcType == DIRICHLET ){
              parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
                var(i,j,k) = (*spec).value;
              });
            } else {
              throw InvalidValue("Error: BC type for table variable not supported for variable: "+varname,__FILE__,__LINE__);
            }


          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
TableLookup::addLookupSpecies( ){

  ChemHelper& helper = ChemHelper::self();
  std::vector<std::string>& sps = helper.model_req_species;
  std::vector<std::string>& old_sps = helper.model_req_old_species;

  std::vector<std::string> missing;
  std::vector<std::string> old_missing;

  for ( auto i_tab = m_tables.begin(); i_tab != m_tables.end(); i_tab++ ){

    int counter = 0;
    for ( auto i = sps.begin(); i != sps.end(); i ++ ){

      bool test = i_tab->second->insertIntoMap( *i );

      if ( test == false ){
        missing.push_back(*i);
      }

      counter++;

    }

    counter = 0;

    for ( auto i = old_sps.begin(); i != old_sps.end(); i ++ ){

      bool test = i_tab->second->insertOldIntoMap( *i );

      if ( test == false ){
        old_missing.push_back(*i);
      }

      counter++;

    }
  }

  if ( missing.size() > 0 ){
    std::stringstream msg;
    msg << " Error: The following species were not found in the table: " << endl;
    for (auto isp = missing.begin(); isp != missing.end(); isp++ ){
      msg << "       " << *isp << endl;
    }
    throw InvalidValue( msg.str(), __FILE__, __LINE__ );
  }

  if ( old_missing.size() > 0 ){
    std::stringstream msg;
    msg << " Error: The following species were not found in the table: " << endl;
    for (auto isp = old_missing.begin(); isp != old_missing.end(); isp++ ){
      msg << "       " << *isp << endl;
    }
    throw InvalidValue( msg.str(), __FILE__, __LINE__ );
  }

}

} //namespace Uintah
