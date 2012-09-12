#include <CCA/Components/Arches/IntrusionBC.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>

#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah; 

//_________________________________________
IntrusionBC::IntrusionBC( const ArchesLabel* lab, const MPMArchesLabel* mpmlab, Properties* props, int WALL ) : 
  _lab(lab), _mpmlab(mpmlab), _props(props), _WALL(WALL)
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

  _intrusion_on = false; 
  _do_energy_exchange = false; 
  _mpm_energy_exchange = false;

}

//_________________________________________
IntrusionBC::~IntrusionBC()
{
  if ( _intrusion_on ) { 
    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){ 

      VarLabel::destroy(iIntrusion->second.bc_area); 
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
IntrusionBC::problemSetup( const ProblemSpecP& params ) 
{

  ProblemSpecP db = params; //<IntrusionBC>

  // The main <intrusion> block lookup
  if ( db->findBlock("intrusion") ){ 

    for ( ProblemSpecP db_intrusion = db->findBlock("intrusion"); db_intrusion != 0; db_intrusion = db_intrusion->findNextBlock("intrusion") ){ 

      Boundary intrusion; 

      std::string name; 
      db_intrusion->getAttribute("label", name);
      intrusion.name = name; 

      // set up velocity:
      ProblemSpecP db_velocity = db_intrusion->findBlock("velocity"); 
      intrusion.has_velocity_model = false; 
      intrusion.mass_flow_rate = 0.0; 

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

        for ( ProblemSpecP db_single_scalar = db_scalars->findBlock("scalar"); 
            db_single_scalar != 0; db_single_scalar = db_single_scalar->findNextBlock("scalar") ){

          std::string scalar_type;
          std::string scalar_label;
          db_single_scalar->getAttribute("type",scalar_type); 
          db_single_scalar->getAttribute("label",scalar_label); 

          scalarInletBase* scalar_bc = 0; 

          if ( scalar_type == "flat" ){ 

            scalar_bc = scinew constantScalar(); 

          }  else if ( scalar_type == "from_file" ){ 

            scalar_bc = scinew scalarFromInput( scalar_label ); 

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
      for ( ProblemSpecP db_labels = db_intrusion->findBlock("variable"); db_labels != 0; db_labels = db_labels->findNextBlock("variable") ){ 

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

      if ( intrusion.type != IntrusionBC::SIMPLE_WALL ) {

        for ( ProblemSpecP db_ds = db_intrusion->findBlock("flux_dir"); 
            db_ds != 0; db_ds = db_ds->findNextBlock("flux_dir") ){ 
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

      //make an area varlable
      intrusion.bc_area = VarLabel::create( name + "_bc_area", sum_vartype::getTypeDescription() ); 

      //this is for the face iterator
      intrusion.has_been_initialized = false; 

      IntrusionMap::iterator i = _intrusion_map.find( name ); 
      if ( i == _intrusion_map.end() ){ 
        _intrusion_map.insert(make_pair( name, intrusion ));  
        _intrusion_on = true; 
      } else { 
        throw ProblemSetupException("Error: Two intrusion boundarys with the same name listed in input file", __FILE__, __LINE__); 
      } 

    } 
  } 
}

//_________________________________________
void 
IntrusionBC::sched_computeBCArea( SchedulerP& sched, 
                                  const PatchSet* patches, 
                                  const MaterialSet* matls )
{

  Task* tsk = scinew Task("IntrusionBC::computeBCArea", this, &IntrusionBC::computeBCArea); 

  for ( IntrusionMap::iterator i = _intrusion_map.begin(); i != _intrusion_map.end(); ++i ){ 

    tsk->computes( i->second.bc_area ); 

  } 

  sched->addTask(tsk, patches, matls); 

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
    //int archIndex = 0; 
    //int index = _lab->d_sharedState->getArchesMaterial( archIndex )->getDWIndex(); 
    Box patch_box = patch->getBox(); 
    Vector Dx = patch->dCell(); 

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){

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

    }   // intrusion loop 
  }     // patch loop
} 

//_________________________________________
void 
IntrusionBC::sched_computeProperties( SchedulerP& sched, 
                                      const PatchSet* patches, 
                                      const MaterialSet* matls )
{
  Task* tsk = scinew Task("IntrusionBC::computeProperties", this, &IntrusionBC::computeProperties); 

  sched->addTask(tsk, patches, matls); 
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
    typedef std::map<std::string, double> StringDoubleMap;

    vector<double> iv; 

    for ( IntrusionMap::iterator iIntrusion = _intrusion_map.begin(); iIntrusion != _intrusion_map.end(); ++iIntrusion ){ 

      if ( !iIntrusion->second.bc_face_iterator.empty() && iIntrusion->second.type != IntrusionBC::SIMPLE_WALL ){ 

        MixingRxnModel* mixingTable = _props->getMixRxnModel(); 
        StringVec iv_var_names = mixingTable->getAllIndepVars(); 

        BCIterator::iterator iBC_iter = (iIntrusion->second.bc_face_iterator).find(patchID); 

        // start face iterator
        for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

          IntVector c = *i; 

          for ( unsigned int niv = 0; niv < iv_var_names.size(); niv++ ){ 

           // iv[niv] = 0.0;

            std::map<std::string, scalarInletBase*>::iterator scalar_iter = iIntrusion->second.scalar_map.find( iv_var_names[niv] ); 

            if ( scalar_iter == iIntrusion->second.scalar_map.end() ){ 
              throw InvalidValue("Error: Cannot compute property values for IntrusionBC. Make sure all IV's are specified!", __FILE__, __LINE__); 
            } 

            double scalar_var = scalar_iter->second->get_scalar( c ); 
            //iv[niv] = scalar_var;
            iv.push_back(scalar_var); 

          }

          double density = mixingTable->getTableValue(iv, "density"); 
          iIntrusion->second.density_map.insert(std::make_pair(c, density)); 
          //
          //Note: Using the last value of density to set the total intrusion density.  
          //This is needed for mass flow inlet conditions but assumes a constant density across the face
          iIntrusion->second.density = density; 

        } // ... end of face iterator ... 
      } 
    }
  }
}

//_________________________________________
void 
IntrusionBC::sched_setIntrusionVelocities( SchedulerP& sched, 
                                  const PatchSet* patches, 
                                  const MaterialSet* matls )
{
  Task* tsk = scinew Task("IntrusionBC::setIntrusionVelocities", this, &IntrusionBC::setIntrusionVelocities); 

  for ( IntrusionMap::iterator i = _intrusion_map.begin(); i != _intrusion_map.end(); ++i ){ 

    tsk->requires( Task::NewDW, i->second.bc_area ); 

  } 

  sched->addTask(tsk, patches, matls); 
}
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
    //Vector Dx = patch->dCell(); 

    //NOTE!  This only works for constant mass flow rates; 

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){ 

      // get the velocity value for the normal component based on total area
      double V = 0.0; 
      sum_vartype area_var; 
      new_dw->get( area_var, iter->second.bc_area ); 
      double area = area_var; 

      if ( iter->second.mass_flow_rate != 0.0 ){ 

        V = iter->second.mass_flow_rate / ( iter->second.density * area ); 

        // now loop through all 6 directions 
        for ( int idir = 0; idir < 6; idir++ ){ 

          if ( iter->second.directions[idir] != 1 ) { 

            int vel_index = _iHelp[idir]; 
            IntVector c = IntVector(0,0,0); 
            iter->second.velocity[vel_index] = V; 
            iter->second.velocity_inlet_generator->massflowrate_velocity( vel_index, V ); 

          }
        } 
      }  // if mass flow rate option has been selected
    }    // intrusion loop 
  }      // patch loop
}

//_________________________________________
void 
IntrusionBC::sched_setCellType( SchedulerP& sched, 
                                  const PatchSet* patches, 
                                  const MaterialSet* matls, 
                                  const bool doing_restart )
{
  Task* tsk = scinew Task("IntrusionBC::setCellType", this, &IntrusionBC::setCellType, doing_restart); 

  if ( !doing_restart ){ 
    tsk->modifies( _lab->d_cellTypeLabel ); 
    tsk->modifies( _lab->d_areaFractionLabel ); 
    tsk->modifies( _lab->d_volFractionLabel ); 
  }
  sched->addTask(tsk, patches, matls); 
}
void 
IntrusionBC::setCellType( const ProcessorGroup*, 
                          const PatchSubset* patches, 
                          const MaterialSubset* matls, 
                          DataWarehouse* old_dw, 
                          DataWarehouse* new_dw, 
                          const bool doing_restart )
{ 
  for ( int p = 0; p < patches->size(); p++ ){ 

    const Patch* patch = patches->get(p); 
    const int patchID = patch->getID(); 
    int archIndex = 0; 
    int index = _lab->d_sharedState->getArchesMaterial( archIndex )->getDWIndex(); 
    Box patch_box = patch->getBox(); 
    //Vector Dx = patch->dCell(); 

    CCVariable<int> cell_type; 
    CCVariable<Vector> area_fraction; 
    CCVariable<double> vol_fraction; 

    if ( !doing_restart ){ 
      new_dw->getModifiable( cell_type, _lab->d_cellTypeLabel, index, patch ); 
      new_dw->getModifiable( area_fraction, _lab->d_areaFractionLabel, index, patch ); 
      new_dw->getModifiable( vol_fraction,  _lab->d_volFractionLabel, index, patch ); 
    }

    for ( IntrusionMap::iterator iter = _intrusion_map.begin(); iter != _intrusion_map.end(); ++iter ){ 

      //make sure cell face iterator map is clean from the start: 
      if ( !iter->second.has_been_initialized ){ 
        iter->second.bc_face_iterator.clear(); 
        iter->second.interior_cell_iterator.clear(); 
        iter->second.has_been_initialized = true; 
      }
      initialize_the_iterators( patchID, iter->second ); 

      for ( int i = 0; i < (int)iter->second.geometry.size(); i++ ){ 

        GeometryPieceP piece = iter->second.geometry[i]; 
        Box geometry_box  = piece->getBoundingBox(); 
        Box intersect_box = geometry_box.intersect( patch_box ); 

        for ( CellIterator icell = patch->getCellIterator(); !icell.done(); icell++ ) { 

          IntVector c = *icell; 

          // check current cell
          // Initialize as a wall
          bool curr_cell = in_or_out( c, piece, patch, iter->second.inverted ); 

          if ( !doing_restart ) { 
            if ( curr_cell ) { 

              cell_type[c] = _WALL; 
              vol_fraction[c] = 0.0; 
              area_fraction[c] = Vector(0.,0.,0.); 

            } else { 

              // this is flow...is the neighbor a solid? 
              // -x direction 
              IntVector n = c - IntVector(1,0,0); 
              bool neighbor_cell = in_or_out( n, piece, patch, iter->second.inverted );  

              Vector af = Vector(1.,1.,1.); 

              if ( neighbor_cell ){
                af -= Vector(1.,0,0); 
              } 
              // -y direciton
              n = c - IntVector(0,1,0); 
              neighbor_cell = in_or_out( n, piece, patch, iter->second.inverted );  

              if ( neighbor_cell ){
                af -= Vector(0,1.,0); 
              } 
              
              // -z direciton
              n = c - IntVector(0,0,1); 
              neighbor_cell = in_or_out( n, piece, patch, iter->second.inverted );  

              if ( neighbor_cell ){
                af -= Vector(0,0,1.); 
              } 
            } 
          }

          for ( int idir = 0; idir < 6; idir++ ){ 

            //check if current cell is in 
            if ( curr_cell ){ 

              if ( iter->second.directions[idir] == 1 ){ 

                IntVector neighbor_index = c + _dHelp[idir]; 
                bool neighbor_cell = in_or_out( neighbor_index, piece, patch, iter->second.inverted ); 

                if ( !neighbor_cell ){ 
                  IntVector face_index = c + _faceDirHelp[idir]; 
                  add_face_iterator( face_index, patch, idir, iter->second ); 
                  add_interior_iterator( neighbor_index, patch, idir, iter->second );
                } 
              } 

            } else { 

              if ( iter->second.directions[idir] == 1 ){ 

                IntVector neighbor_index = c - _dHelp[idir];
                bool neighbor_cell = in_or_out( neighbor_index, piece, patch, iter->second.inverted ); 

                if ( neighbor_cell ){ 

                  IntVector face_index = neighbor_index + _faceDirHelp[idir]; 
                  add_face_iterator( face_index, patch, idir, iter->second ); 
                  add_interior_iterator( c, patch, idir, iter->second ); 

                } 
              } 
            } 
          } 
        } 
      } // geometry loop

      //debugging: 
//      std::cout << " ======== " << std::endl;
//      std::cout << " For Intrusion named: " << iter->first << std::endl;
//      print_iterator( patchID, iter->second ); 
      

      // For this collection of  geometry pieces, the iterator is now established.  
      // loop through and repair all the relevant area fractions using 
      // the new boundary face iterator. 
      if ( !doing_restart ) { 
        if ( !(iter->second.bc_face_iterator.empty()) ){

          BCIterator::iterator iBC_iter = (iter->second.bc_face_iterator).find(patchID); 

          for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); 
              i != iBC_iter->second.end(); i++){

            IntVector c = *i; 

            for ( int idir = 0; idir < 6; idir++ ){ 

              if ( iter->second.directions[idir] == 1 ){ 

                if ( patch->containsCell( c ) ){ 
                
                  area_fraction[c][_iHelp[idir]] = 0; 

                } 
              }
            }
          }
        }
      }
    }   // intrusion loop 
  }     // patch loop
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

      if ( !iIntrusion->second.bc_face_iterator.empty() ) {

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
        if ( scalar_iter == iIntrusion->second.scalar_map.end() ){ 
          throw InvalidValue("Error: Cannot match scalar value to scalar name in intrusion: "+scalar_name, __FILE__, __LINE__); 
        } 

        if ( !iIntrusion->second.interior_cell_iterator.empty() ) {

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
        if ( scalar_iter == iIntrusion->second.scalar_map.end() ){ 
          throw InvalidValue("Error: Cannot match scalar value to scalar name in intrusion: "+scalar_name, __FILE__, __LINE__); 
        } 

        if ( !iIntrusion->second.interior_cell_iterator.empty() ) {

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

        if ( !iIntrusion->second.interior_cell_iterator.empty() ) {

          BCIterator::iterator  iBC_iter = (iIntrusion->second.interior_cell_iterator).find(p);

          for ( std::vector<IntVector>::iterator i = iBC_iter->second.begin(); i != iBC_iter->second.end(); i++){

            IntVector c = *i;

            for ( int idir = 0; idir < 6; idir++ ){ 

              if ( iIntrusion->second.directions[idir] != 0 ){ 

                density[ c - _faceDirHelp[idir] ] = iIntrusion->second.density; 

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
                                  const PatchSet* patches, 
                                  const MaterialSet* matls )
{ 
  if ( _do_energy_exchange ){ 
    Task* tsk = scinew Task("IntrusionBC::setIntrusionT", this, &IntrusionBC::setIntrusionT); 

    _T_label = VarLabel::find("temperature"); 

    tsk->modifies( _T_label );  

    if ( _mpmlab && _mpm_energy_exchange ){ 
      tsk->requires( Task::NewDW, _mpmlab->integTemp_CCLabel, Ghost::None, 0 );  
    } 

    sched->addTask( tsk, patches, matls );
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

