#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <ostream>
#include <fstream>
#include <stdlib.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName):
d_fieldLabels(fieldLabels), d_timeIntegrator(timeIntegrator), d_eqnName(eqnName),
b_stepUsesCellLocation(false), b_stepUsesPhysicalLocation(false),
d_constant_init(0.0), d_step_dir("x"), d_step_start(0.0), d_step_end(0.0), d_step_cellstart(0), d_step_cellend(0), d_step_value(0.0), 
d_use_constant_D(false)
{
  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels->d_sharedState->getArchesMaterial(0)->getDWIndex() ); 
  d_disc = scinew Discretization_new(); 
  _using_new_intrusion = false; 
  _table_init = false; 
  _stage = 1;  //uses density after first table lookup  
}

EqnBase::~EqnBase()
{
  delete(d_boundaryCond);
  delete(d_disc);
}

void
EqnBase::extraProblemSetup( ProblemSpecP& db ){ 

  d_boundaryCond->setupTabulatedBC( db, d_eqnName, _table );

}

void 
EqnBase::commonProblemSetup( ProblemSpecP& db ){ 

  // Clipping:
  // defaults: 
  clip.activated = false;
  clip.do_low  = false; 
  clip.do_high = false; 
  clip.my_type = ClipInfo::STANDARD;

  ProblemSpecP db_clipping = db->findBlock("Clipping");

  if (db_clipping) {

    std::string type = "default"; 
    std::fstream inputFile;
    std::string clip_dep_file; 
    std::string clip_dep_low_file;   
    std::string clip_ind_file;      

    if ( db_clipping->getAttribute( "type", type )){

      db_clipping->getAttribute( "type", type );
      if ( type == "variable_constrained" ){

        clip.my_type = ClipInfo::CONSTRAINED;
        db_clipping->findBlock("constraint")->getAttribute("label", clip.ind_var );

        db_clipping->require("clip_dep_file",clip_dep_file);
        db_clipping->require("clip_dep_low_file",clip_dep_low_file);        
        db_clipping->require("clip_ind_file",clip_ind_file);
  
        // read clipping file   
        double number;
        inputFile.open(clip_dep_file.c_str());
        while (inputFile >> number)
        {
          clip_dep_vec.push_back(number);
        }
        inputFile.close(); 
  
        inputFile.open(clip_dep_low_file.c_str());
        while (inputFile >> number)
        {
          clip_dep_low_vec.push_back(number);
        }
        inputFile.close(); 
  
  
        // read clipping file   
        inputFile.open(clip_ind_file.c_str());
        while (inputFile >> number)
        {
          clip_ind_vec.push_back(number);
        }
        inputFile.close(); 
      
      }
    
    } else { 

      clip.activated = true; 
      
      db_clipping->getWithDefault("low", clip.low,  -1.e16);
      db_clipping->getWithDefault("high",clip.high, 1.e16);
      db_clipping->getWithDefault("tol", clip.tol, 1e-10); 

      if ( db_clipping->findBlock("low") ) 
        clip.do_low = true; 

      if ( db_clipping->findBlock("high") ) 
        clip.do_high = true;  

      if ( !clip.do_low && !clip.do_high ) 
        throw InvalidValue("Error: A low or high clipping must be specified if the <Clipping> section is activated.", __FILE__, __LINE__);

    }

  } 

  // Initialization:
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {

    db_initialValue->getAttribute("type", d_initFunction); 

    if (d_initFunction == "constant") {
      db_initialValue->require("constant", d_constant_init); 

    } else if (d_initFunction == "step") {
      db_initialValue->require("step_direction", d_step_dir); 
      db_initialValue->require("step_value", d_step_value); 

      if( db_initialValue->findBlock("step_start") ) {
        b_stepUsesPhysicalLocation = true;
        db_initialValue->require("step_start", d_step_start); 
        db_initialValue->require("step_end"  , d_step_end); 

      } else if ( db_initialValue->findBlock("step_cellstart") ) {
        b_stepUsesCellLocation = true;
        db_initialValue->require("step_cellstart", d_step_cellstart);
        db_initialValue->require("step_cellend", d_step_cellend);
      }

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here. 
    } else if (d_initFunction == "geometry_fill") {

      db_initialValue->require("constant_inside", d_constant_in_init);              //fill inside geometry
      db_initialValue->getWithDefault( "constant_outside",d_constant_out_init,0.0); //fill outside geometry 

      ProblemSpecP the_geometry = db_initialValue->findBlock("geom_object"); 
      if (the_geometry) {
        GeometryPieceFactory::create(the_geometry, d_initGeom); 
      } else {
        throw ProblemSetupException("You are missing the geometry specification (<geom_object>) for the transport eqn. initialization!", __FILE__, __LINE__); 
      }
    } else if ( d_initFunction == "gaussian" ) { 

      db_initialValue->require( "amplitude", d_a_gauss ); 
      db_initialValue->require( "center", d_b_gauss ); 
      db_initialValue->require( "std", d_c_gauss ); 
      std::string direction; 
      db_initialValue->require( "direction", direction ); 
      if ( direction == "X" || direction == "x" ){ 
        d_dir_gauss = 0; 
      } else if ( direction == "Y" || direction == "y" ){ 
        d_dir_gauss = 1; 
      } else if ( direction == "Z" || direction == "z" ){
        d_dir_gauss = 2; 
      } 
      db_initialValue->getWithDefault( "shift", d_shift_gauss, 0.0 ); 

    } else if ( d_initFunction == "tabulated" ){ 

      db_initialValue->require( "depend_varname", d_init_dp_varname ); 
      _table_init = true; 
      
    } else if ( d_initFunction == "shunn_moin"){ 

      ProblemSpecP db_mom = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("MomentumSolver"); 
      string init_type;
      db_mom->findBlock("initialization")->getAttribute("type",init_type); 
      if ( init_type != "shunn_moin"){ 
        throw InvalidValue("Error: Trying to initialize the Shunn/Moin MMS for the mixture fraction and not matching same IC in momentum", __FILE__,__LINE__); 
      }
      db_mom->findBlock("initialization")->require("k",d_k);
      db_mom->findBlock("initialization")->require("w",d_w); 
      std::string plane;
      db_mom->findBlock("initialization")->require("plane",plane); 

      ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow"); 
      
      db_prop->findBlock("stream_0")->getAttribute("density",d_rho0);
      db_prop->findBlock("stream_1")->getAttribute("density",d_rho1);

      if ( plane == "x-y"){ 
        d_dir0=0;
        d_dir1=1;
      } else if ( plane == "y-z" ){ 
        d_dir0=1;
        d_dir1=2;
      } else if ( plane == "z-x" ){ 
        d_dir0=2;
        d_dir1=0;
      } else {
        throw InvalidValue("Error: Plane not recognized. Please choose xy, yz, or xz",__FILE__,__LINE__);
      }

    }
  }

  // Molecular diffusivity: 
  d_use_constant_D = false; 
  if ( db->findBlock( "D_mol" ) ){ 
    db->findBlock("D_mol")->getAttribute("label", d_mol_D_label_name); 
  } else if ( db->findBlock( "D_mol_constant" ) ){ 
    db->findBlock("D_mol_constant")->getAttribute("value", d_mol_diff ); 
    d_use_constant_D = true; 
  } else { 
    d_use_constant_D = true; 
    d_mol_diff = 0.0; 
    proc0cout << "NOTICE: For equation " << d_eqnName << " no molecular diffusivity was specified.  Assuming D=0.0. \n";
  } 

  if ( db->findBlock( "D_mol" ) && db->findBlock( "D_mol_constant" ) ){ 
    string err_msg = "ERROR: For transport equation: "+d_eqnName+" \n Molecular diffusivity is over specified. \n";
    throw ProblemSetupException(err_msg, __FILE__, __LINE__); 
  } 
}

void 
EqnBase::sched_checkBCs( const LevelP& level, SchedulerP& sched )
{
  string taskname = "EqnBase::checkBCs"; 
  Task* tsk = scinew Task(taskname, this, &EqnBase::checkBCs); 
  
   // We want tasks in boundarycondition.cc to run before this task. We use the following dummy label to achieve this.
   const VarLabel* DummyLabel = VarLabel::find("ForceTaskExecutionOrder");
   tsk->requires(Task::NewDW, DummyLabel, Ghost::None,0 ); 
    
  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() ); 
}

void 
EqnBase::checkBCs( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);
    double dx=0; 
    double dy=0; 

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){
      Patch::FaceType face = *bf_iter; 

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        Vector bc_v_value(0,0,0); 
        std::string bc_s_value = "NA";

        Iterator bound_ptr;
        string bc_kind = "NotSet"; 
        string face_name; 
        
        
        getBCKind( patch, face, child, d_eqnName, matlIndex, bc_kind, face_name ); 

        std::ofstream outputfile; 
        std::stringstream fname; 
        fname << "handoff_" << d_eqnName << "_" << face_name <<  "." << patch->getID();
        bool file_is_open = false; 

        string whichface; 
        int index=0; 
        Vector Dx = patch->dCell(); 

        if (face == 0){
          whichface = "x-";
          index = 0;
          dx = Dx[1];
          dy = Dx[2];
        } else if (face == 1) {
          whichface = "x+"; 
          index = 0;
          dx = Dx[1];
          dy = Dx[2];
        } else if (face == 2) { 
          whichface = "y-";
          index = 1;
          dx = Dx[2];
          dy = Dx[0];
        } else if (face == 3) {
          whichface = "y+";
          index = 1;
          dx = Dx[2];
          dy = Dx[0];
        } else if (face == 4) {
          whichface = "z-";
          index = 2;
          dx = Dx[0];
          dy = Dx[1];
        } else if (face == 5) {
          whichface = "z+";
          index = 2;
          dx = Dx[0];
          dy = Dx[1];
        }

        if ( bc_kind == "NotSet" ){ 

          cout << "ERROR!:  Missing boundary condition specification!" << endl;
          cout << "Here are the details:" << endl;
          cout << "Variable = " << d_eqnName << endl;
          cout << "Face = " << whichface << endl; 
          cout << "Child = " << child << endl;
          cout << "Material = " << matlIndex << endl;
          throw ProblemSetupException("Please correct your <BoundaryCondition> section in your input file for this variable", __FILE__,__LINE__); 
        }

        // need to map x,y,z -> i,j,k for the FromFile option
        bool foundIterator = false; 
        if ( bc_kind == "FromFile" ){ 
          foundIterator = 
            getIteratorBCValue<std::string>( patch, face, child, d_eqnName, matlIndex, bc_s_value, bound_ptr ); 
        } 

        BoundaryCondition_new::ScalarToBCValueMap& scalar_bc_info = d_boundaryCond->get_FromFileInfo(); 
        BoundaryCondition_new::ScalarToBCValueMap::iterator i_scalar_bc_storage = scalar_bc_info.find( face_name ); 

        //check the grid spacing: 
        if ( i_scalar_bc_storage != scalar_bc_info.end() ){ 
          proc0cout <<  endl << "For scalar handoff file named: " << i_scalar_bc_storage->second.name << endl;
          proc0cout <<          "  Grid and handoff spacing relative differences are: [" 
            << std::abs(i_scalar_bc_storage->second.dx - dx)/dx << ", " 
            << std::abs(i_scalar_bc_storage->second.dy - dy)/dy << "]" << endl << endl;
        }

        if (foundIterator) {

          //if we are here, then we are of type "FromFile" 
          bound_ptr.reset(); 

          //this should assign the correct normal direction xyz value without forcing the user to have 
          //to know what it is. 
          if ( index == 0 ) { 
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          } else if ( index == 1 ) { 
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          } else if ( index == 2 ) { 
            i_scalar_bc_storage->second.relative_xyz[index] = Dx[index]/2.0;
          } 
          Vector ref_point = i_scalar_bc_storage->second.relative_xyz;
          Point xyz(ref_point[0],ref_point[1],ref_point[2]);

          IntVector ijk = patch->getLevel()->getCellIndex( xyz ); 

          i_scalar_bc_storage->second.relative_ijk = ijk;
          i_scalar_bc_storage->second.relative_ijk[index] = 0;  //don't allow the normal index to shift

          int face_index_value=10;

          //now check to make sure that there is a bc set for each iterator: 
          for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){ 
            //The next three lines are needed because we are ignoring the user input 
            //for the normal index but still loading it into memory
            IntVector mod_bound_ptr = (*bound_ptr);
            face_index_value = mod_bound_ptr[index]; 
            mod_bound_ptr[index] = (i_scalar_bc_storage->second.values.begin()->first)[index];
            BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.find(mod_bound_ptr - i_scalar_bc_storage->second.relative_ijk);
            if ( check_iter == i_scalar_bc_storage->second.values.end() ){ 
              std::stringstream out; 
              out <<  "Scalar BC: " << d_eqnName << " - No UINTAH boundary cell " << *bound_ptr - i_scalar_bc_storage->second.relative_ijk << " in the handoff file." << endl;
              if ( !file_is_open ){ 
                file_is_open = true; 
                outputfile.open(fname.str().c_str());
                outputfile << "Patch Dimentions (exclusive): \n";
                outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                outputfile << " high = " << patch->getCellHighIndex() << "\n";
                outputfile << out.str();  
              } else { 
                outputfile << out.str();  
              } 
            } 
          } 

          //now check the reverse -- does the handoff file have an associated boundary ptr
          BoundaryCondition_new::CellToValueMap temp_map; 
          for ( BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.begin(); check_iter != 
              i_scalar_bc_storage->second.values.end(); check_iter++ ){ 

            //need to reset the values to get the right [index] int value for the face
            double value = check_iter->second; 
            IntVector location = check_iter->first;
            location[index] = face_index_value; 

            temp_map.insert(make_pair(location, value)); 

          }

          //reassign the values now with the correct index for the face direction
          i_scalar_bc_storage->second.values = temp_map; 

          for ( BoundaryCondition_new::CellToValueMap::iterator check_iter = i_scalar_bc_storage->second.values.begin(); check_iter != 
              i_scalar_bc_storage->second.values.end(); check_iter++ ){ 

            bool found_it = false; 
            for ( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ){ 
              if ( *bound_ptr == (check_iter->first + i_scalar_bc_storage->second.relative_ijk) )
                found_it = true; 
            }
            if ( !found_it && patch->containsCell(check_iter->first + i_scalar_bc_storage->second.relative_ijk) ){ 
              std::stringstream out; 
              out << "Scalar BC: " << d_eqnName << " - No HANDOFF cell " << check_iter->first << " (relative) in the Uintah geometry object." << endl;
              if ( !file_is_open ){ 
                file_is_open = true;
                outputfile.open(fname.str().c_str());
                outputfile << "Patch Dimentions (exclusive): \n";
                outputfile << " low  = " << patch->getCellLowIndex() << "\n";
                outputfile << " high = " << patch->getCellHighIndex() << "\n";
                outputfile << out.str();  
              } else { 
                outputfile << out.str();  
              } 
            } 
          } 

        }
        if ( file_is_open ){ 
          cout << "\n  Notice: Handoff scalar " << d_eqnName << " has warning information printed to file for patch #: " << patch->getID() << "\n"; 
          outputfile.close(); 
        } 
      }
    }
  }
}

void
EqnBase::sched_tableInitialization( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "EqnBase::tableInitialization";
  Task* tsk = scinew Task(taskname, this, &EqnBase::tableInitialization); 

  MixingRxnModel::VarMap ivVarMap = _table->getIVVars();

  // independent variables :: these must have been computed previously 
  for ( MixingRxnModel::VarMap::iterator i = ivVarMap.begin(); i != ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, Ghost::None, 0 ); 

  }

  // for inert mixing
  MixingRxnModel::InertMasterMap inertMap = _table->getInertMap(); 
  for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){ 
    const VarLabel* label = VarLabel::find( iter->first ); 
    tsk->requires( Task::NewDW, label, Ghost::None, 0 ); 
  } 

  tsk->modifies( d_transportVarLabel ); 

  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() ); 


}

void 
EqnBase::tableInitialization(const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage; 
    MixingRxnModel::VarMap ivVarMap = _table->getIVVars();
    std::vector<string> allIndepVarNames = _table->getAllIndepVars(); 

    for ( int i = 0; i < (int) allIndepVarNames.size(); i++ ){

      MixingRxnModel::VarMap::iterator ivar = ivVarMap.find( allIndepVarNames[i] ); 

      constCCVariable<double> the_var; 
      new_dw->get( the_var, ivar->second, matlIndex, patch, Ghost::None, 0 );
      indep_storage.push_back( the_var ); 

    }

    MixingRxnModel::InertMasterMap inertMap = _table->getInertMap(); 
    MixingRxnModel::StringToCCVar inert_mixture_fractions; 
    inert_mixture_fractions.clear(); 
    for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){ 
      const VarLabel* label = VarLabel::find( iter->first ); 
      constCCVariable<double> variable; 
      new_dw->get( variable, label, matlIndex, patch, Ghost::None, 0 ); 
      MixingRxnModel::ConstVarContainer container; 
      container.var = variable; 

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) ); 

    } 

    CCVariable<double> eqn_var; 
    new_dw->getModifiable( eqn_var, d_transportVarLabel, matlIndex, patch ); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      std::vector<double> iv;
      for (std::vector<constCCVariable<double> >::iterator iv_iter = indep_storage.begin(); 
          iv_iter != indep_storage.end(); iv_iter++ ){ 

        iv.push_back( (*iv_iter)[c] ); 

      }

      eqn_var[c] = _table->getTableValue( iv, d_init_dp_varname, inert_mixture_fractions, c ); 

    }

    //recompute the BCs
    computeBCsSpecial( patch, d_eqnName, eqn_var );

  }
}
