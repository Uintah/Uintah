#include <CCA/Components/Arches/PropertyModelsV2/sootVolumeFrac.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
////Method: Constructor
////---------------------------------------------------------------------------
sootVolumeFrac::sootVolumeFrac( std::string task_name, int matl_index ) : TaskInterface( task_name, matl_index)
{
}
  
  

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
sootVolumeFrac::~sootVolumeFrac( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 

}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void sootVolumeFrac::problemSetup(  Uintah::ProblemSpecP& db )
{
  db->getAttribute("label", _fvSoot);
  db->getWithDefault( "density_label" ,          _den_label_name,    "density"	  );
  db->getWithDefault( "Ysoot_label"   ,          _Ys_label_name ,    "Ysoot"	  );
  db->getWithDefault( "soot_density", _rho_soot, 1950.0);
}





void
sootVolumeFrac::create_local_labels(){
  register_new_variable<CCVariable<double> >(_fvSoot);

}



void
sootVolumeFrac::register_initialize( VIVec& variable_registry , const bool pack_tasks){
  register_variable( _fvSoot, Uintah::ArchesFieldContainer::COMPUTES, variable_registry );
}

void
sootVolumeFrac::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){
  CCVariable<double>& fvSoot     = tsk_info->get_uintah_field_add<CCVariable<double> >( _fvSoot);
  fvSoot.initialize(0.0);
}

void sootVolumeFrac::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

}

void sootVolumeFrac::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void sootVolumeFrac::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

}


void sootVolumeFrac::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void
sootVolumeFrac::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){
  register_variable( _fvSoot, Uintah::ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable(_den_label_name , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::LATEST,variable_registry, time_substep );
  register_variable(_Ys_label_name , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::LATEST,variable_registry, time_substep );
}


void
sootVolumeFrac::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& fvSoot     = tsk_info->get_uintah_field_add<CCVariable<double> >( _fvSoot);
  fvSoot.initialize(0.0);

  constCCVariable<double>& gas_density= tsk_info->get_const_uintah_field_add<constCCVariable<double> >( _den_label_name);
  constCCVariable<double>& Ysoot= tsk_info->get_const_uintah_field_add<constCCVariable<double> >( _Ys_label_name);


    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for(range,  [&](int i, int j, int k) {
        fvSoot(i,j,k) = gas_density(i,j,k) * Ysoot(i,j,k) / _rho_soot;
      });
}



