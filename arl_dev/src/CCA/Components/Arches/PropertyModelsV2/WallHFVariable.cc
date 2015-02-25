#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <spatialops/structured/FVStaggered.h>

using namespace Uintah;

WallHFVariable::WallHFVariable( std::string task_name, int matl_index, SimulationStateP& shared_state ) : 
TaskInterface( task_name, matl_index ) { 

  _flux_x = task_name + "_x"; 
  _flux_y = task_name + "_y"; 
  _flux_z = task_name + "_z";

  _shared_state = shared_state; 

}

WallHFVariable::~WallHFVariable(){ 
}

void 
WallHFVariable::problemSetup( ProblemSpecP& db ){ 

  db->getWithDefault("frequency",_f,1);

}

void 
WallHFVariable::create_local_labels(){ 

  register_new_variable( _flux_x, CC_DOUBLE ); 
  register_new_variable( _flux_y, CC_DOUBLE ); 
  register_new_variable( _flux_z, CC_DOUBLE ); 

}

//
//------------------------------------------------
//-------------- INITIALIZATION ------------------
//------------------------------------------------
//

void 
WallHFVariable::register_initialize( std::vector<VariableInformation>& variable_registry ){ 

  register_variable( _flux_x, CC_DOUBLE, COMPUTES, variable_registry ); 
  register_variable( _flux_y, CC_DOUBLE, COMPUTES, variable_registry ); 
  register_variable( _flux_z, CC_DOUBLE, COMPUTES, variable_registry ); 

}

void 
WallHFVariable::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){ 

  using namespace SpatialOps;
  using SpatialOps::operator *; 
  typedef SpatialOps::SVolField   SVolF;
  typedef SpatialOps::SpatFldPtr<SVolF> SVolFP; 

  SVolFP flux_x = tsk_info->get_so_field<SVolF>(_flux_x); 
  SVolFP flux_y = tsk_info->get_so_field<SVolF>(_flux_y); 
  SVolFP flux_z = tsk_info->get_so_field<SVolF>(_flux_z); 

  *flux_x <<= 0.0;
  *flux_y <<= 0.0;
  *flux_z <<= 0.0;

}

//
//------------------------------------------------
//------------- TIMESTEP WORK --------------------
//------------------------------------------------
//

void 
WallHFVariable::register_timestep_eval( std::vector<VariableInformation>& variable_registry, const int time_substep ){ 

  register_variable( _flux_x            , CC_DOUBLE , COMPUTES , variable_registry );
  register_variable( _flux_y            , CC_DOUBLE , COMPUTES , variable_registry );
  register_variable( _flux_z            , CC_DOUBLE , COMPUTES , variable_registry );
  register_variable( "radiationFluxE"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "radiationFluxW"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "radiationFluxN"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "radiationFluxS"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "radiationFluxT"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "radiationFluxB"   , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( "volFraction"      , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( _flux_x            , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( _flux_y            , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );
  register_variable( _flux_z            , CC_DOUBLE , REQUIRES , 0 , OLDDW , variable_registry );

}

void 
WallHFVariable::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                SpatialOps::OperatorDatabase& opr ){ 

  constCCVariable<double>* Fe = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxE"); 
  constCCVariable<double>* Fw = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxW"); 
  constCCVariable<double>* Fn = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxN"); 
  constCCVariable<double>* Fs = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxS"); 
  constCCVariable<double>* Ft = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxT"); 
  constCCVariable<double>* Fb = tsk_info->get_uintah_const_field<constCCVariable<double> >("radiationFluxB"); 
  constCCVariable<double>* volFraction = tsk_info->get_uintah_const_field<constCCVariable<double> >("volFraction"); 
  constCCVariable<double>* old_flux_x = tsk_info->get_uintah_const_field<constCCVariable<double> >(_flux_x); 
  constCCVariable<double>* old_flux_y = tsk_info->get_uintah_const_field<constCCVariable<double> >(_flux_y); 
  constCCVariable<double>* old_flux_z = tsk_info->get_uintah_const_field<constCCVariable<double> >(_flux_z); 

  CCVariable<double>* flux_x = tsk_info->get_uintah_field<CCVariable<double> >(_flux_x); 
  CCVariable<double>* flux_y = tsk_info->get_uintah_field<CCVariable<double> >(_flux_y); 
  CCVariable<double>* flux_z = tsk_info->get_uintah_field<CCVariable<double> >(_flux_z); 

  (*flux_x).initialize(0.0); 
  (*flux_y).initialize(0.0); 
  (*flux_z).initialize(0.0); 

  int timestep = _shared_state->getCurrentTopLevelTimeStep(); 

  //if ( ( timestep )%_f + 1 == 1 ){ 
  if ( ( timestep )%_f  == 0 ){ 

    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 

      IntVector c = *iter; 
      IntVector cxp = c + IntVector(1,0,0); 
      IntVector cxm = c - IntVector(1,0,0); 
      IntVector cyp = c + IntVector(0,1,0); 
      IntVector cym = c - IntVector(0,1,0); 
      IntVector czp = c + IntVector(0,0,1); 
      IntVector czm = c - IntVector(0,0,1); 

      if ( (*volFraction)[c] > 0.0 ){ 

        ////check neighbors to see if we populate a flux here: 
        if ( (*volFraction)[cxm] < 1.0 ){ 
          (*flux_x)[c] = (*flux_x)[c] + (*Fw)[c];
        }
        if ( (*volFraction)[cxp] < 1.0 ){ 
          (*flux_x)[c] = (*flux_x)[c] + (*Fe)[c];
        }
        if ( (*volFraction)[cym] < 1.0 ){ 
          (*flux_y)[c] = (*flux_y)[c] + (*Fs)[c];
        }
        if ( (*volFraction)[cxp] < 1.0 ){ 
          (*flux_y)[c] = (*flux_y)[c] + (*Fn)[c];
        }
        if ( (*volFraction)[czm] < 1.0 ){ 
          (*flux_z)[c] = (*flux_z)[c] + (*Fb)[c];
        }
        if ( (*volFraction)[czp] < 1.0 ){ 
          (*flux_z)[c] = (*flux_z)[c] + (*Ft)[c];
        }

      }
    }
  } else { 

    (*flux_x).copyData((*old_flux_x));
    (*flux_y).copyData((*old_flux_y));
    (*flux_z).copyData((*old_flux_z));

  }
  
}
