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

#include <CCA/Components/Arches/IntrusionBC.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>

using namespace Uintah;

namespace {

// These are for uniquely identifying the Uintah::CrowdMonitors<Tag>
// used to protect multi-threaded access to global data structures
struct intrustion_map_tag{};
using  intrusion_map_monitor = Uintah::CrowdMonitor<intrustion_map_tag>;

using Mutex = Uintah::MasterLock;
Mutex intrusion_print_mutex{};

}

//_________________________________________
IntrusionBC::IntrusionBC( const ArchesLabel* lab, const MPMArchesLabel* mpmlab, Properties* props,
                          TableLookup* table_lookup, int WALL )
  : _lab(lab)
  , _mpmlab(mpmlab)
  , _props(props)
  , _table_lookup(table_lookup)
  , _WALL(WALL)
{
  // helper for the intvector direction
  _dHelp.push_back( IntVector(-1,0,0) );
  _dHelp.push_back( IntVector(+1,0,0) );
  _dHelp.push_back( IntVector(0,-1,0) );
  _dHelp.push_back( IntVector(0,+1,0) );
  _dHelp.push_back( IntVector(0,0,-1) );
  _dHelp.push_back( IntVector(0,0,+1) );

  // helper for the indexing for face cells
  _faceDirHelp.push_back( IntVector(0,0,0) );
  _faceDirHelp.push_back( IntVector(+1,0,0) );
  _faceDirHelp.push_back( IntVector(0,0,0) );
  _faceDirHelp.push_back( IntVector(0,+1,0) );
  _faceDirHelp.push_back( IntVector(0,0,0) );
  _faceDirHelp.push_back( IntVector(0,0,+1) );

  // helper for getting neighboring interior cell
  // neighbor = face_iter + _inside[direction];
  _inside.push_back( IntVector(-1,0,0) );
  _inside.push_back( IntVector( 0,0,0) );
  _inside.push_back( IntVector( 0,-1,0) );
  _inside.push_back( IntVector( 0,0,0) );
  _inside.push_back( IntVector( 0,0,-1) );
  _inside.push_back( IntVector( 0,0,0) );

  // helper for referencing the right index depending on direction
  _iHelp.push_back( 0 );
  _iHelp.push_back( 0 );
  _iHelp.push_back( 1 );
  _iHelp.push_back( 1 );
  _iHelp.push_back( 2 );
  _iHelp.push_back( 2 );

  // helper for the sign on the face
  _sHelp.push_back( -1.0 );
  _sHelp.push_back( +1.0 );
  _sHelp.push_back( -1.0 );
  _sHelp.push_back( +1.0 );
  _sHelp.push_back( -1.0 );
  _sHelp.push_back( +1.0 );

  _intrusion_on        = false;
  _do_energy_exchange  = false;
  _mpm_energy_exchange = false;

}

//_________________________________________
IntrusionBC::~IntrusionBC()
{

  delete localPatches_;

  if ( _intrusion_on ) {
    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      if ( (iIntrusion->second).type == INLET ){
        VarLabel::destroy(iIntrusion->second.bc_area);
      }

      if ( iIntrusion->second.has_velocity_model )  {
        delete(iIntrusion->second.velocity_inlet_generator);
      }

      for ( std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.begin();
          scalar_iter != iIntrusion->second.scalar_map.end(); scalar_iter++ ){

        delete(scalar_iter->second);

      }
    }
  }
}

//_________________________________________
void
IntrusionBC::problemSetup( const ProblemSpecP& params, const int ilvl )
{
  ProblemSpecP db = params; //<IntrusionBC>

  // The main <intrusion> block lookup
  if( db->findBlock("intrusion") ) {

    for( ProblemSpecP db_intrusion = db->findBlock( "intrusion" ); db_intrusion != nullptr; db_intrusion = db_intrusion->findNextBlock( "intrusion" ) ) {

      Boundary intrusion;

      std::string name;
      db_intrusion->getAttribute("label", name);
      intrusion.name = name;
      intrusion.velocity = Vector(0,0,0);

      if ( db_intrusion->findBlock("ignore_missing_bc") ){
        intrusion.ignore_missing_bc = true;
      } else {
        intrusion.ignore_missing_bc = false;
      }

      // set up velocity:
      ProblemSpecP db_velocity = db_intrusion->findBlock("velocity");
      intrusion.has_velocity_model = false;
      intrusion.mass_flow_rate = 0.0;
      intrusion.velocity_inlet_generator = 0;

      if ( db_velocity ){

        intrusion.has_velocity_model = true;

        std::string vel_type;
        db_velocity->getAttribute("type",vel_type);

        if ( vel_type == "flat" ){

          intrusion.type = IntrusionBC::INLET;
          intrusion.velocity_inlet_generator = scinew FlatVelProf();

        } else if ( vel_type == "from_file" ){

          intrusion.type = IntrusionBC::INLET;
          intrusion.velocity_inlet_generator = scinew InputFileVelocity();

        } else if ( vel_type == "massflow" ){

          intrusion.type = IntrusionBC::INLET;
          intrusion.velocity_inlet_generator = scinew FlatVelProf();

          double flow_rate = 0.0;
          db_intrusion->findBlock("velocity")->getWithDefault("flow_rate",flow_rate, 0.0);
          intrusion.mass_flow_rate = flow_rate;

        } else {

          throw ProblemSetupException("Error: Invalid <velocity> type attribute for intrusion "+name,__FILE__,__LINE__);

        }

        intrusion.velocity_inlet_generator->problem_setup( db_intrusion );


      } else {

        intrusion.type = IntrusionBC::SIMPLE_WALL;

      }

      // set up scalars:
      ProblemSpecP db_scalars = db_intrusion->findBlock("scalars");
      if ( db_scalars ){

        for ( ProblemSpecP db_single_scalar = db_scalars->findBlock("scalar"); db_single_scalar != nullptr; db_single_scalar = db_single_scalar->findNextBlock("scalar") ){

          std::string scalar_type;
          std::string scalar_label;
          db_single_scalar->getAttribute("type",scalar_type);
          db_single_scalar->getAttribute("label",scalar_label);

          scalarInletBase* scalar_bc = 0;

          if ( scalar_type == "flat" ){

            scalar_bc = scinew constantScalar();

          }  else if ( scalar_type == "from_file" ){

            scalar_bc = scinew scalarFromInput( scalar_label );

          } else if ( scalar_type == "tabulated" ){

            scalar_bc = scinew tabulatedScalar();

          } else {

            throw ProblemSetupException("Error: Invalid intrusion <scalar> type attribute. ",__FILE__,__LINE__);

          }

          scalar_bc->problem_setup( db_single_scalar, db_intrusion );

          intrusion.scalar_map.insert(make_pair( scalar_label, scalar_bc ));

        }
      }

      intrusion.inverted = false;
      if ( db_intrusion->findBlock("inverted") ){
        intrusion.inverted = true;
      }

      //geometry
      ProblemSpecP geometry_db = db_intrusion->findBlock("geom_object");
      GeometryPieceFactory::create( geometry_db, intrusion.geometry );

      //labels
      for ( ProblemSpecP db_labels = db_intrusion->findBlock("variable"); db_labels != nullptr; db_labels = db_labels->findNextBlock("variable") ){

        std::string label_name;
        double label_value;

        db_labels->getAttribute( "label", label_name );
        db_labels->getAttribute( "value", label_value );

        intrusion.varnames_values_map.insert(make_pair(label_name, label_value));

      }

      //direction of boundary
      //initialize to zero
      std::vector<int> temp;
      for (int i = 0; i<6; ++i ){
        temp.push_back(0);
      }
      intrusion.directions = temp;

      if ( intrusion.type == IntrusionBC::INLET ) {

        for ( ProblemSpecP db_ds = db_intrusion->findBlock("flux_dir"); db_ds != nullptr; db_ds = db_ds->findNextBlock("flux_dir") ){
          std::string my_dir;
          my_dir = db_ds->getNodeValue();
          if ( my_dir == "x-" || my_dir == "X-"){

            intrusion.directions[0] = 1;

          } else if ( my_dir == "x+" || my_dir == "X+"){

            intrusion.directions[1] = 1;

          } else if ( my_dir == "y-" || my_dir == "Y-"){

            intrusion.directions[2] = 1;

          } else if ( my_dir == "y+" || my_dir == "Y+"){

            intrusion.directions[3] = 1;

          } else if ( my_dir == "z-" || my_dir == "Z-"){

            intrusion.directions[4] = 1;

          } else if ( my_dir == "z+" || my_dir == "Z+"){

            intrusion.directions[5] = 1;

          } else {
            proc0cout << "Warning: Intrusion flux direction = " << my_dir << " not recognized.  Ignoring...\n";
          }
        }

        //make an area varlable
        std::string level_index = std::to_string(ilvl);
        intrusion.bc_area = VarLabel::create( name + "_bc_area_"+level_index, sum_vartype::getTypeDescription() );

      }

      //temperature of the intrusion
      // Either choose constant T or an integrated T from MPM
      intrusion.temperature = 298.0;
      if ( db_intrusion->findBlock( "constant_temperature" ) ){
        db_intrusion->findBlock("constant_temperature")->getAttribute("T", intrusion.temperature);
        _do_energy_exchange = true;
      }
      if ( db_intrusion->findBlock( "mpm_temperature" ) ){
        if ( _do_energy_exchange ){
          throw ProblemSetupException("Error: Cannot specify both <constant_temperature> and <mpm_temperature>.", __FILE__, __LINE__);
        }
        _do_energy_exchange = true;
      }

      //initialize density
      intrusion.density = 0.0;

      //this is for the face iterator
      intrusion.has_been_initialized = false;

      IntrusionMap::iterator i = _intrusion_map.find(name);
      if (i == _intrusion_map.end()) {
        _intrusion_map.insert(make_pair(name, intrusion));
        _intrusion_on = true;
      } else {
        throw ProblemSetupException("Error: Two intrusion boundaries with the same name listed in input file", __FILE__, __LINE__);
      }

    }
  }
}

//_________________________________________
void
IntrusionBC::sched_computeBCArea( SchedulerP& sched,
                                  const LevelP& level,
                                  const MaterialSet* matls )
{

  Task* tsk = scinew Task("IntrusionBC::computeBCArea", this, &IntrusionBC::computeBCArea);

  bool found_inlet_intrusion = false;
  for ( IntrusionMap::iterator i = _intrusion_map.begin(); i != _intrusion_map.end(); ++i ){

    if ( (i->second).type == INLET ){
      tsk->computes( i->second.bc_area );
      found_inlet_intrusion = true;
    }

  }

  if ( found_inlet_intrusion ){
    sched->addTask(tsk, level->eachPatch(), matls);
  } else {
    delete tsk;
  }

}
void
IntrusionBC::computeBCArea( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{

  for ( int p = 0; p < patches->size(); p++ ){

    const Patch* patch = patches->get(p);
    Box patch_box = patch->getBox();
    Vector Dx = patch->dCell();

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){

      if ( (iter->second).type == INLET ){

        double total_area = 0.;

        for ( int i = 0; i < (int)iter->second.geometry.size(); i++ ){

          GeometryPieceP piece = iter->second.geometry[i];
          Box geometry_box  = piece->getBoundingBox();
          Box intersect_box = geometry_box.intersect( patch_box );

          if ( !(intersect_box.degenerate()) ) {

            for ( CellIterator icell = patch->getCellIterator(); !icell.done(); icell++ ) {

              IntVector c = *icell;

              // now loop through all 6 directions
              for ( int idir = 0; idir < 6; idir++ ){

                if ( iter->second.directions[idir] != 0 ) {
                  double darea;
                  if ( idir == 0 || idir == 1 ) {
                    darea = Dx.y()*Dx.z();
                  } else if ( idir == 2 || idir == 3 ) {
                    darea = Dx.x()*Dx.z();
                  } else {
                    darea = Dx.x()*Dx.y();
                  }

                  // check current cell:
                  bool curr_cell = in_or_out( c, piece, patch, iter->second.inverted );

                  if ( curr_cell ){
                    //check neighbor in the direction of the boundary
                    IntVector n = c + _dHelp[idir];
                    bool neighbor_cell = in_or_out( n, piece, patch, iter->second.inverted );

                    if ( !neighbor_cell ) {

                      total_area += darea;

                    }
                  }
                }
              }
            }
          }
        } // geometry loop

        new_dw->put( sum_vartype( total_area ), iter->second.bc_area );

      } // if INLET

    }   // intrusion loop
  }     // patch loop
}

//_________________________________________
void
IntrusionBC::sched_computeProperties( SchedulerP& sched,
                                      const LevelP& level,
                                      const MaterialSet* matls )
{
  Task* tsk = scinew Task("IntrusionBC::computeProperties", this, &IntrusionBC::computeProperties);

  sched->addTask(tsk, level->eachPatch(), matls);
}

void
IntrusionBC::computeProperties( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw )
{
  for ( int p = 0; p < patches->size(); p++ ){

    const Patch* patch = patches->get(p);
    const int patchID = patch->getID();

    typedef std::vector<std::string> StringVec;
    std::vector<double> iv;

    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      if ( !iIntrusion->second.bc_cell_iterator.empty() && iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){

        MixingRxnModel* mixingTable = _table_lookup->get_table();
        StringVec iv_var_names = mixingTable->getAllIndepVars();

        BCIterator::iterator iBC_iter = (iIntrusion->second.bc_cell_iterator).find(patchID);

        // start face iterator
        bool found_valid_density = false;
        double found_density = 0.0;

        for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

          IntVector c = *i;
          iv.clear();

          cout_intrusiondebug << "IntrusionBC::For Intrusion named: " << iIntrusion->second.name << std::endl;
          cout_intrusiondebug << "IntrusionBC::At location = " << c << std::endl;

          for ( unsigned int niv = 0; niv < iv_var_names.size(); niv++ ){

           // iv[niv] = 0.0;

            std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( iv_var_names[niv] );

            if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
              throw InvalidValue("Error: Cannot compute property values for IntrusionBC. Make sure all IV's are specified!", __FILE__, __LINE__);
            }

            double scalar_var = scalar_iter->second->get_scalar( c );
            //iv[niv] = scalar_var;
            iv.push_back(scalar_var);

            cout_intrusiondebug << "IntrusionBC::For independent variable " << iv_var_names[niv] << ". Using value = " << scalar_var << std::endl;

          }

          bool does_post_mix = mixingTable->doesPostMix();

          double density = 0.0;
          typedef std::map<std::string, double> DMap;
          DMap inert_list;


          if ( does_post_mix ){

            cout_intrusiondebug << "IntrusionBC::Using inert stream mixing to look up properties" << std::endl;

            typedef std::map<std::string, DMap > IMap;
            IMap inert_map = mixingTable->getInertMap();
            for ( IMap::iterator imap =  inert_map.begin();
                                 imap != inert_map.end(); imap++ ){
              std::string name = imap->first;
              std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( name );

              if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
                throw InvalidValue("Error: Cannot compute property values for IntrusionBC. Make sure all participating inerts are specified!", __FILE__, __LINE__);
              }

              double inert_value = scalar_iter->second->get_scalar( c );
              inert_list.insert(std::make_pair(name,inert_value));

              cout_intrusiondebug << "IntrusionBC::For inert variable " << name << ". Using value = " << inert_value << std::endl;

            }

            density = mixingTable->getTableValue(iv, "density",inert_list);

            cout_intrusiondebug << "IntrusionBC::Got a value for density = " << density << std::endl;

            //get values for all other scalars that depend on a table lookup:
            for (std::map<std::string, scalarInletBase*>::iterator iter_lookup = iIntrusion->second.scalar_map.begin();
                                                                   iter_lookup != iIntrusion->second.scalar_map.end();
                                                                   iter_lookup++ ){

              if ( iter_lookup->second->get_type() == scalarInletBase::TABULATED ){

                tabulatedScalar& tab_scalar = dynamic_cast<tabulatedScalar&>(*iter_lookup->second);

                std::string lookup_name = tab_scalar.get_depend_var_name();

                double lookup_value = mixingTable->getTableValue(iv, lookup_name,inert_list);

                cout_intrusiondebug << "IntrusionBC::Setting scalar " << iter_lookup->first << " to a lookup value of: " << lookup_value << std::endl;

                tab_scalar.set_scalar_constant( lookup_value );

              }

            }

          } else {

            cout_intrusiondebug << "IntrusionBC::NOT using inert stream mixing to look up properties" << std::endl;

            density = mixingTable->getTableValue(iv, "density");

            //get values for all other scalars that depend on a table lookup:
            for (std::map<std::string, scalarInletBase*>::iterator iter_lookup = iIntrusion->second.scalar_map.begin();
                                                                   iter_lookup != iIntrusion->second.scalar_map.end();
                                                                   iter_lookup++ ){

              if ( iter_lookup->second->get_type() == scalarInletBase::TABULATED ){

                tabulatedScalar& tab_scalar = dynamic_cast<tabulatedScalar&>(*iter_lookup->second);

                std::string lookup_name = tab_scalar.get_depend_var_name();

                double lookup_value = mixingTable->getTableValue(iv, lookup_name);

                cout_intrusiondebug << "IntrusionBC::Setting scalar " << iter_lookup->first << " to a lookup value of: " << lookup_value << std::endl;

                tab_scalar.set_scalar_constant( lookup_value );

              }

            }
          }

          iIntrusion->second.density_map.insert(std::make_pair(c, density));
          //
          //Note: Using the last value of density to set the total intrusion density.
          //This is needed for mass flow inlet conditions but assumes a constant density across the face
          if ( std::abs(density) > 1e-10 ){
            found_density = density;
            found_valid_density = true;
          }

        } // ... end of face iterator ...

        if ( found_valid_density ){
          iIntrusion->second.density = found_density;
        }
      }
      if ( !iIntrusion->second.bc_face_iterator.empty() && iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){

        MixingRxnModel* mixingTable = _table_lookup->get_table();
        StringVec iv_var_names = mixingTable->getAllIndepVars();

        BCIterator::iterator iBC_iter = (iIntrusion->second.bc_face_iterator).find(patchID);

        // start face iterator
        bool found_valid_density = false;
        double found_density = 0.0;

        for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

          IntVector c = *i;
          iv.clear();

          cout_intrusiondebug << "IntrusionBC::For Intrusion named: " << iIntrusion->second.name << std::endl;
          cout_intrusiondebug << "IntrusionBC::At location = " << c << std::endl;

          for ( unsigned int niv = 0; niv < iv_var_names.size(); niv++ ){

           // iv[niv] = 0.0;

            std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( iv_var_names[niv] );

            if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
              throw InvalidValue("Error: Cannot compute property values for IntrusionBC. Make sure all IV's are specified!", __FILE__, __LINE__);
            }

            double scalar_var = scalar_iter->second->get_scalar( c );
            //iv[niv] = scalar_var;
            iv.push_back(scalar_var);

            cout_intrusiondebug << "IntrusionBC::For independent variable " << iv_var_names[niv] << ". Using value = " << scalar_var << std::endl;

          }

          bool does_post_mix = mixingTable->doesPostMix();

          double density = 0.0;
          typedef std::map<std::string, double> DMap;
          DMap inert_list;


          if ( does_post_mix ){

            cout_intrusiondebug << "IntrusionBC::Using inert stream mixing to look up properties" << std::endl;

            typedef std::map<std::string, DMap > IMap;
            IMap inert_map = mixingTable->getInertMap();
            for ( IMap::iterator imap =  inert_map.begin();
                                 imap != inert_map.end(); imap++ ){
              std::string name = imap->first;
              std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( name );

              if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
                throw InvalidValue("Error: Cannot compute property values for IntrusionBC. Make sure all participating inerts are specified!", __FILE__, __LINE__);
              }

              double inert_value = scalar_iter->second->get_scalar( c );
              inert_list.insert(std::make_pair(name,inert_value));

              cout_intrusiondebug << "IntrusionBC::For inert variable " << name << ". Using value = " << inert_value << std::endl;

            }

            density = mixingTable->getTableValue(iv, "density",inert_list);

            cout_intrusiondebug << "IntrusionBC::Got a value for density = " << density << std::endl;

            //get values for all other scalars that depend on a table lookup:
            for (std::map<std::string, scalarInletBase*>::iterator iter_lookup = iIntrusion->second.scalar_map.begin();
                                                                   iter_lookup != iIntrusion->second.scalar_map.end();
                                                                   iter_lookup++ ){

              if ( iter_lookup->second->get_type() == scalarInletBase::TABULATED ){

                tabulatedScalar& tab_scalar = dynamic_cast<tabulatedScalar&>(*iter_lookup->second);

                std::string lookup_name = tab_scalar.get_depend_var_name();

                double lookup_value = mixingTable->getTableValue(iv, lookup_name,inert_list);

                cout_intrusiondebug << "IntrusionBC::Setting scalar " << iter_lookup->first << " to a lookup value of: " << lookup_value << std::endl;

                tab_scalar.set_scalar_constant( lookup_value );

              }

            }

          } else {

            cout_intrusiondebug << "IntrusionBC::NOT using inert stream mixing to look up properties" << std::endl;

            density = mixingTable->getTableValue(iv, "density");

            //get values for all other scalars that depend on a table lookup:
          }

          iIntrusion->second.density_map.insert(std::make_pair(c, density));
          //
          //Note: Using the last value of density to set the total intrusion density.
          //This is needed for mass flow inlet conditions but assumes a constant density across the face
          if ( std::abs(density) > 1e-10 ){
            found_density = density;
            found_valid_density = true;
          }

        } // ... end of face iterator ...

        if ( found_valid_density ){
          iIntrusion->second.density = found_density;
        }
      }
    }
  }
}

//_________________________________________
void
IntrusionBC::sched_setIntrusionVelocities( SchedulerP& sched,
                                           const LevelP& level,
                                           const MaterialSet* matls )
{
  Task* tsk = scinew Task("IntrusionBC::setIntrusionVelocities", this, &IntrusionBC::setIntrusionVelocities);

  bool found_inlet_intrusion = false;
  for ( IntrusionMap::iterator i = _intrusion_map.begin(); i != _intrusion_map.end(); ++i ){

    if ( (i->second).type == INLET ){
      tsk->requires( Task::NewDW, i->second.bc_area );
      found_inlet_intrusion = true;
    }

  }

  if ( found_inlet_intrusion ){
    sched->addTask(tsk, level->eachPatch(), matls);
  } else {
    delete tsk;
  }
}

//_________________________________________
void
IntrusionBC::setIntrusionVelocities( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  for ( int p = 0; p < patches->size(); p++ ){

    const Patch* patch = patches->get(p);
    Box patch_box = patch->getBox();

    //NOTE!  This only works for constant mass flow rates;

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){

      if ( iter->second.type == INLET ){

        // get the velocity value for the normal component based on total area
        double V = 0.0;
        sum_vartype area_var;
        new_dw->get( area_var, iter->second.bc_area );
        double area = area_var;

        if ( iter->second.mass_flow_rate != 0.0 ){

          V = iter->second.mass_flow_rate / ( iter->second.density * area );

          // now loop through all 6 directions
          for ( int idir = 0; idir < 6; idir++ ){

            if ( iter->second.directions[idir] == 1 ) {

              int vel_index = _iHelp[idir];
              IntVector c = IntVector(0,0,0);
              iter->second.velocity[vel_index] = V;
              iter->second.velocity_inlet_generator->massflowrate_velocity( vel_index, V );

            }
          }
        }  // if mass flow rate option has been selected
      }    // intrusion loop
    }      // if INLET
  }        // patch loop
}

//_________________________________________
void
IntrusionBC::sched_setCellType( SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* matls,
                                const bool doing_restart )
{
  Task* tsk = scinew Task("IntrusionBC::setCellType", this, &IntrusionBC::setCellType, doing_restart);

  if ( !doing_restart ){
    tsk->modifies( _lab->d_cellTypeLabel );
    tsk->modifies( _lab->d_areaFractionLabel );
    tsk->modifies( _lab->d_volFractionLabel );
  }
  sched->addTask(tsk, level->eachPatch(), matls);
}

//_________________________________________
void
IntrusionBC::setCellType( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const bool doing_restart )
{
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const int patchID = patch->getID();
    int archIndex = 0;
    int index = _lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    Box patch_box = patch->getBox();

    CCVariable<int> cell_type;
    CCVariable<Vector> area_fraction;
    CCVariable<double> vol_fraction;

    if (!doing_restart) {
      new_dw->getModifiable(cell_type, _lab->d_cellTypeLabel, index, patch);
      new_dw->getModifiable(area_fraction, _lab->d_areaFractionLabel, index, patch);
      new_dw->getModifiable(vol_fraction, _lab->d_volFractionLabel, index, patch);
    }

    // Scope the multi-reader CrowdMonitor
    {
      intrusion_map_monitor intrusion_map_lock { intrusion_map_monitor::WRITER };

      for (IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter) {

        //make sure cell face iterator map is clean from the start:
        if (!iter->second.has_been_initialized) {
          iter->second.bc_face_iterator.clear();
          iter->second.interior_cell_iterator.clear();
          iter->second.bc_cell_iterator.clear();
          iter->second.has_been_initialized = true;
        }

        // ----------------------------------------------------------------------
        // NOTE: the inline method initialize_the_iterators (below) has been made thread-safe due to shared data structures:
        //  BCIterator - typedef std::map<int, std::vector<IntVector> >
        //  bc_face_iterator, interior_cell_iterator and bc_cell_iterator
        // ----------------------------------------------------------------------
        // These data structures are used in member functions, of which some are scheduled tasks, e.g.:
        //  IntrusionBC::setHattedVelocity, IntrusionBC::addScalarRHS,
        //  IntrusionBC::setDensity, IntrusionBC::computeProperties
        // ----------------------------------------------------------------------
        initialize_the_iterators(patchID, iter->second);

        for (int i = 0; i < (int)iter->second.geometry.size(); i++) {

          GeometryPieceP piece = iter->second.geometry[i];
          Box geometry_box = piece->getBoundingBox();
          Box intersect_box = geometry_box.intersect(patch_box);

          for (CellIterator icell = patch->getCellIterator(); !icell.done(); icell++) {

            IntVector c = *icell;

            // check current cell
            // Initialize as a wall
            bool curr_cell = in_or_out(c, piece, patch, iter->second.inverted);

            if (!doing_restart) {
              if (curr_cell) {

                cell_type[c] = _WALL;
                vol_fraction[c] = 0.0;
                area_fraction[c] = Vector(0., 0., 0.);

              } else {

                // this is flow...is the neighbor a solid?
                // -x direction
                IntVector n = c - IntVector(1, 0, 0);
                bool neighbor_cell = in_or_out(n, piece, patch, iter->second.inverted);

                Vector af = Vector(1., 1., 1.);

                if (neighbor_cell) {
                  af -= Vector(1., 0, 0);
                }
                // -y direction
                n = c - IntVector(0, 1, 0);
                neighbor_cell = in_or_out(n, piece, patch, iter->second.inverted);

                if (neighbor_cell) {
                  af -= Vector(0, 1., 0);
                }

                // -z direction
                n = c - IntVector(0, 0, 1);
                neighbor_cell = in_or_out(n, piece, patch, iter->second.inverted);

                if (neighbor_cell) {
                  af -= Vector(0, 0, 1.);
                }
              }
            }

            // ----------------------------------------------------------------------
            // NOTE: the inline methods: add_face_iterator, add_interior_iterator and add_bc_cell_iterator (below)
            //  have been made thread-safe due to shared data structures:
            //  BCIterator - typedef std::map<int, std::vector<IntVector> >
            //  bc_face_iterator, interior_cell_iterator and bc_cell_iterator
            // ----------------------------------------------------------------------
            // These data structures are used in member functions, of which some are scheduled tasks, e.g.:
            //  IntrusionBC::setHattedVelocity, IntrusionBC::addScalarRHS,
            //  IntrusionBC::setDensity, IntrusionBC::computeProperties
            // ----------------------------------------------------------------------
            for (int idir = 0; idir < 6; idir++) {

              //check if current cell is in
              if (curr_cell) {

                if (iter->second.directions[idir] == 1) {

                  IntVector neighbor_index = c + _dHelp[idir];
                  bool neighbor_cell = in_or_out(neighbor_index, piece, patch, iter->second.inverted);

                  if (!neighbor_cell) {
                    IntVector face_index = c + _faceDirHelp[idir];
                    //face iterator is the face index using the usual convention
                    //note that face iterator + _inside[dir] gives the first wall cell in that direction
                    add_face_iterator(face_index, patch, idir, iter->second);
                    //interior iterator is the first flow cell next to the wall
                    add_interior_iterator(neighbor_index, patch, idir, iter->second);
                    //last wall cell next to outlet
                    add_bc_cell_iterator(c, patch, idir, iter->second);
                  }
                }

              } else {

                if (iter->second.directions[idir] == 1) {

                  IntVector neighbor_index = c - _dHelp[idir];
                  bool neighbor_cell = in_or_out(neighbor_index, piece, patch, iter->second.inverted);

                  if (neighbor_cell) {

                    IntVector face_index = neighbor_index + _faceDirHelp[idir];
                    add_face_iterator(face_index, patch, idir, iter->second);
                    add_interior_iterator(c, patch, idir, iter->second);
                    add_bc_cell_iterator(neighbor_index, patch, idir, iter->second);

                  }
                }
              }
            }
          }
        }  // geometry loop

        //debugging:
        //std::cout << " ======== " << std::endl;
        //std::cout << " For Intrusion named: " << iter->first << std::endl;
        //print_iterator( patchID, iter->second );

        // For this collection of  geometry pieces, the iterator is now established.
        // loop through and repair all the relevant area fractions using
        // the new boundary face iterator.
        if (!doing_restart) {
          if (!(iter->second.bc_face_iterator.empty())) {

            BCIterator::iterator iBC_iter = (iter->second.bc_face_iterator).find(patchID);

            for (std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++) {

              IntVector c = *i;

              for (int idir = 0; idir < 6; idir++) {

                if (iter->second.directions[idir] == 1) {

                  if (patch->containsCell(c)) {

                    area_fraction[c][_iHelp[idir]] = 0;

                  }
                }
              }
            }
          }
        }
      }   // intrusion loop
    }
  }     // patch loop
}

//_________________________________________
void
IntrusionBC::sched_printIntrusionInformation( SchedulerP& sched,
                                              const LevelP& level,
                                              const MaterialSet* matls )
{

  Task* tsk = scinew Task("IntrusionBC::printIntrusionInformation", this, &IntrusionBC::printIntrusionInformation);

  for ( IntrusionMap::iterator i = _intrusion_map.begin(); i != _intrusion_map.end(); ++i ){

    if ( i->second.type == INLET ){
      tsk->requires( Task::NewDW, i->second.bc_area );
    }

  }

  sched->addTask(tsk, level->eachPatch(), matls);

}
void
IntrusionBC::printIntrusionInformation( const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw )
{
  // RAII-style approach to acquiring output mutex for this entire scoped block.
  std::lock_guard<Mutex> print_lock(intrusion_print_mutex);

  for (int p = 0; p < patches->size(); p++) {

    proc0cout << "----- Intrusion Summary ----- \n " << std::endl;

    for (IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter) {

      double area = 0.;
      if (iter->second.type == SIMPLE_WALL) {

        proc0cout << " Intrusion name/type: " << iter->first << " / Simple wall " << std::endl;

      } else if (iter->second.type == INLET) {

        sum_vartype area_var;
        new_dw->get(area_var, iter->second.bc_area);
        area = area_var;

        proc0cout << " Intrusion name/type: " << iter->first << " / Inlet" << std::endl;
        proc0cout << "           density  = " << iter->second.density << std::endl;
        proc0cout << "         inlet area = " << area << std::endl << std::endl;

        proc0cout << " Active inlet directions (normals): " << std::endl;

        for (int idir = 0; idir < 6; idir++) {

          if (iter->second.directions[idir] != 0) {
            proc0cout << "   " << _dHelp[idir] << std::endl;
          }

        }

        proc0cout << std::endl << " Scalar information: " << std::endl;

        for (std::map<std::string, scalarInletBase*>::iterator i_scalar = iter->second.scalar_map.begin();
            i_scalar != iter->second.scalar_map.end(); i_scalar++) {

          IntVector c(0, 0, 0);
          proc0cout << "     -> " << i_scalar->first << ":   value = " << i_scalar->second->get_scalar(c) << std::endl;

        }

      }

      proc0cout << std::endl;

      proc0cout << " Solid T  = " << iter->second.temperature << std::endl;

      proc0cout << " \n";

    }
    proc0cout << "----- End Intrusion Summary ----- \n " << std::endl;
  }
}

//_________________________________________
void
IntrusionBC::setHattedVelocity( const Patch*  patch,
                                SFCXVariable<double>& u,
                                SFCYVariable<double>& v,
                                SFCZVariable<double>& w,
                                constCCVariable<double>& density )
{
  // go through each intrusion
  // go through the iterator for this patch
  // set the velocities according to method chosen in input file
  // exit
  const int p = patch->getID();

  if ( _intrusion_on ) {

    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      BCIterator::iterator  iBC_iter = (iIntrusion->second.bc_face_iterator).find(p);

      for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

        IntVector c = *i;

        for ( int idir = 0; idir < 6; idir++ ){

          if ( iIntrusion->second.directions[idir] != 0 ){

            iIntrusion->second.velocity_inlet_generator->set_velocity( idir, c, u, v, w, density,
                iIntrusion->second.density );

          }
        }
      }
    }
  }
}

//_________________________________________
void
IntrusionBC::setScalar( const int p,
                        const std::string scalar_name,
                        CCVariable<double>& scalar ){

  std::cout << " ERROR!  DANGER WILL ROBINSON!" << std::endl;
  throw InvalidValue("Error: IntrusionBC::setScalar not implemented ", __FILE__, __LINE__);
//  if ( _intrusion_on ) {
//
//    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){
//
//      std::map<std::string,double>::iterator scalar_iter =  iIntrusion->second.varnames_values_map.find( scalar_name );
//
//      if ( scalar_iter == iIntrusion->second.varnames_values_map.end() ){
//        throw InvalidValue("Error: Cannot match scalar value to scalar name in intrusion. ", __FILE__, __LINE__);
//      }
//
//      if ( !iIntrusion->second.bc_face_iterator.empty() ) {
//        BCIterator::iterator  iBC_iter = (iIntrusion->second.bc_face_iterator).find(p);
//
//        for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){
//
//          //scalar[*i] = scalar_iter->second;
//
//        }
//      }
//    }
//  }
}

//_________________________________________
void
IntrusionBC::addScalarRHS( const Patch* patch,
                           Vector Dx,
                           const std::string scalar_name,
                           CCVariable<double>& RHS
                           )
{
  const int p = patch->getID();
  std::vector<double> area;
  area.push_back(Dx.y()*Dx.z());
  area.push_back(Dx.y()*Dx.z());
  area.push_back(Dx.x()*Dx.z());
  area.push_back(Dx.x()*Dx.z());
  area.push_back(Dx.y()*Dx.x());
  area.push_back(Dx.y()*Dx.x());

  if ( _intrusion_on ) {

    // adds \rho*u*\phi to the RHS of the cell NEXT to the boundary
    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      if ( iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){

        //std::map<std::string,double>::iterator scalar_iter =  iIntrusion->second.varnames_values_map.find( scalar_name );
        std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( scalar_name );

        //if ( scalar_iter == iIntrusion->second.varnames_values_map.end() ){
        bool found_bc = true;
        if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
          if ( !iIntrusion->second.ignore_missing_bc )
            throw InvalidValue("Error: Cannot match scalar value to scalar name in intrusion: "+scalar_name, __FILE__, __LINE__);
          found_bc = false;
        }

        if ( !iIntrusion->second.interior_cell_iterator.empty() && found_bc ) {

          BCIterator::iterator  iBC_iter = (iIntrusion->second.interior_cell_iterator).find(p);

          for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

            IntVector c = *i;

            for ( int idir = 0; idir < 6; idir++ ){

              if ( iIntrusion->second.directions[idir] != 0 ){

                double face_den = 1.0;

                const Vector V = iIntrusion->second.velocity_inlet_generator->get_velocity(c);

                double face_vel = V[_iHelp[idir]];

                scalar_iter->second->set_scalar_rhs( idir, c, RHS, face_den, face_vel, area );

              }
            }
          }
        }
      }
    }
  }
}

//_________________________________________
void
IntrusionBC::addScalarRHS( const Patch* patch,
                           Vector Dx,
                           const std::string scalar_name,
                           CCVariable<double>& RHS,
                           constCCVariable<double>& density )
{
  const int p = patch->getID();
  std::vector<double> area;
  area.push_back(Dx.y()*Dx.z());
  area.push_back(Dx.y()*Dx.z());
  area.push_back(Dx.x()*Dx.z());
  area.push_back(Dx.x()*Dx.z());
  area.push_back(Dx.y()*Dx.x());
  area.push_back(Dx.y()*Dx.x());

  if ( _intrusion_on ) {

    // adds \rho*u*\phi to the RHS of the cell NEXT to the boundary
    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      if ( iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){

        //std::map<std::string,double>::iterator scalar_iter =  iIntrusion->second.varnames_values_map.find( scalar_name );
        std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( scalar_name );

        //if ( scalar_iter == iIntrusion->second.varnames_values_map.end() ){
        bool found_bc = true;
        if ( scalar_iter == iIntrusion->second.scalar_map.end() ){
          if ( !iIntrusion->second.ignore_missing_bc )
            throw InvalidValue("Error: Cannot match scalar value to scalar name in intrusion: "+scalar_name, __FILE__, __LINE__);
          found_bc = false;
        }

        if ( !iIntrusion->second.interior_cell_iterator.empty() && found_bc ) {

          BCIterator::iterator  iBC_iter = (iIntrusion->second.interior_cell_iterator).find(p);

          for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

            IntVector c = *i;

            for ( int idir = 0; idir < 6; idir++ ){

              if ( iIntrusion->second.directions[idir] != 0 ){

                double face_den = iIntrusion->second.density;

                const Vector V = iIntrusion->second.velocity_inlet_generator->get_velocity(c);

                double face_vel = V[_iHelp[idir]];

                scalar_iter->second->set_scalar_rhs( idir, c, RHS, face_den, face_vel, area );

              }
            }
          }
        }
      }
    }
  }
}

//_________________________________________
void
IntrusionBC::setDensity( const Patch* patch,
                         CCVariable<double>& density )
{
  const int p = patch->getID();

  if ( _intrusion_on ) {

    // sets density on intrusion inlets
    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){

      if ( iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){

        if ( !iIntrusion->second.bc_cell_iterator.empty() ) {

          BCIterator::iterator  iBC_iter = (iIntrusion->second.bc_cell_iterator).find(p);

          for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

            IntVector c = *i;

            for ( int idir = 0; idir < 6; idir++ ){

              if ( iIntrusion->second.directions[idir] != 0 ){

                density[ c ] = 2.0*iIntrusion->second.density - density[c+_dHelp[idir]];

              }
            }
          }
        }
      }
    }
  }
}

void
IntrusionBC::sched_setIntrusionT( SchedulerP& sched,
                                  const LevelP& level,
                                  const MaterialSet* matls )
{
  if ( _do_energy_exchange ){
    Task* tsk = scinew Task("IntrusionBC::setIntrusionT", this, &IntrusionBC::setIntrusionT);

    _T_label = VarLabel::find("temperature");

    tsk->modifies( _T_label );
    tsk->modifies( _lab->d_densityCPLabel );

    if ( _mpmlab && _mpm_energy_exchange ){
      tsk->requires( Task::NewDW, _mpmlab->integTemp_CCLabel, Ghost::None, 0 );
    }

    sched->addTask( tsk, level->eachPatch(), matls );
  }
}

void
IntrusionBC::setIntrusionT( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )

{
  for ( int p = 0; p < patches->size(); p++ ){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int index = _lab->d_sharedState->getArchesMaterial( archIndex )->getDWIndex();
    Box patch_box = patch->getBox();

    CCVariable<double> temperature;
    new_dw->getModifiable( temperature, _T_label, index, patch );

    constCCVariable<double> mpm_temperature;
    if ( _mpmlab && _mpm_energy_exchange ){
      new_dw->get( mpm_temperature, _mpmlab->integTemp_CCLabel, index, patch, Ghost::None, 0 );
    }

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){

      for ( int i = 0; i < (int)iter->second.geometry.size(); i++ ){

        GeometryPieceP piece = iter->second.geometry[i];
        Box geometry_box  = piece->getBoundingBox();
        Box intersect_box = geometry_box.intersect( patch_box );

        if ( !(intersect_box.degenerate()) ) {

          if ( _mpm_energy_exchange ){
            for ( CellIterator icell = patch->getCellCenterIterator(intersect_box); !icell.done(); icell++ ) {

              IntVector c = *icell;

              // check current cell
              bool curr_cell = in_or_out( c, piece, patch, iter->second.inverted );

              if ( curr_cell ) {

                temperature[c] = mpm_temperature[c];

              }
            }
          } else {
            for ( CellIterator icell = patch->getCellCenterIterator(intersect_box); !icell.done(); icell++ ) {

              IntVector c = *icell;

              // check current cell
              bool curr_cell = in_or_out( c, piece, patch, iter->second.inverted );

              if ( curr_cell ) {

                temperature[c] = iter->second.temperature;

              }
            }
          }
        }
      }
    }
  }
}

//----------------------------------
void
IntrusionBC::prune_per_patch_intrusions( SchedulerP& sched, const LevelP& level, const MaterialSet* matls )
{

  const Uintah::PatchSet* const allPatches =
    sched->getLoadBalancer()->getPerProcessorPatchSet(level);
  const Uintah::PatchSubset* const localPatches =
    allPatches->getSubset( Uintah::Parallel::getMPIRank() );
  localPatches_ = new Uintah::PatchSet;
  localPatches_->addEach( localPatches->getVector() );
  auto mypatches = localPatches->getVector();
  std::vector<std::string> intrusion_map_idx;

  for ( IntrusionMap::iterator i_intrusion = _intrusion_map.begin();
          i_intrusion != _intrusion_map.end(); ++i_intrusion ){

    bool patch_geom_intersection = false;

    for( auto ipatches = (mypatches).begin(); ipatches != mypatches.end(); ipatches++ ){

      std::vector<Patch::FaceType>::const_iterator bf_iter;
      std::vector<Patch::FaceType> bf;
      (*ipatches)->getBoundaryFaces(bf);
      Box patch_box = (*ipatches)->getBox();

      for ( int i = 0; i < (int)i_intrusion->second.geometry.size(); i++ ){

        //Buffer the search region by one cell so as not to miss inlets on patch boundaries.
        GeometryPieceP piece = i_intrusion->second.geometry[i];
        Point low((*ipatches)->cellPosition((*ipatches)->getCellLowIndex()-IntVector(1,1,1)));
        Point high((*ipatches)->cellPosition((*ipatches)->getCellHighIndex()+IntVector(1,1,1)));
        Box geometry_box(low, high);
        Box intersect_box = geometry_box.intersect( patch_box );

        if ( !(intersect_box.degenerate()) ) {
          patch_geom_intersection = true;
        }

      }
    } // patch loop

    // add non-intersecting geometry to a list
    if ( !patch_geom_intersection ){
      intrusion_map_idx.push_back(i_intrusion->first);
    }

  } // intrusion loop

  // now delete the intrusions that aren't resident on this patch
  for ( auto it = intrusion_map_idx.begin(); it != intrusion_map_idx.end(); ++it ){

    if ( _intrusion_map[*it].type == INLET ){
      VarLabel::destroy(_intrusion_map[*it].bc_area);
    }
    if ( _intrusion_map[*it].has_velocity_model )  {
      delete(_intrusion_map[*it].velocity_inlet_generator);
    }

    for ( std::map<std::string, scalarInletBase*>::iterator scalar_iter = _intrusion_map[*it].scalar_map.begin();
        scalar_iter != _intrusion_map[*it].scalar_map.end(); scalar_iter++ ){
      delete(scalar_iter->second);
    }

    _intrusion_map.erase(*it);

  }
}
