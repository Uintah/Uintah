#include <CCA/Components/Arches/PropertyModelsV2/gasRadProperties.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
////Method: Constructor
////---------------------------------------------------------------------------
gasRadProperties::gasRadProperties( std::string task_name, int matl_index ) : TaskInterface( task_name, matl_index)
{
}
  
  

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
gasRadProperties::~gasRadProperties( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 

  delete _calc; 
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void gasRadProperties::problemSetup(  Uintah::ProblemSpecP& db )
{

  ChemHelper& helper = ChemHelper::self();

  db->getAttribute("label",_abskg_name);

  std::string calculator_type; 
  db->getWithDefault("absorption_modifier",_absorption_modifier,1.0);
  ProblemSpecP db_calc = db->findBlock("calculator"); 
  if ( db_calc != nullptr){ 
    db_calc->getAttribute("type",calculator_type); 
  } else { 
    throw InvalidValue("Error: Calculator type not specified.",__FILE__, __LINE__); 
  }

  if ( calculator_type == "constant" ){ 
    _calc = scinew RadPropertyCalculator::ConstantProperties(); 
  } else if ( calculator_type == "special" ){ 
    _calc = scinew RadPropertyCalculator::specialProperties(); 
  } else if ( calculator_type == "burns_christon" ){ 
    _calc = scinew RadPropertyCalculator::BurnsChriston(); 
  } else if ( calculator_type == "hottel_sarofim"){
    _calc = scinew RadPropertyCalculator::HottelSarofim(); 
    helper.add_lookup_species("CO2");
    helper.add_lookup_species("H2O");
  } else if ( calculator_type == "radprops" ){
#ifdef HAVE_RADPROPS
    helper.add_lookup_species("mixture_molecular_weight");
    helper.add_lookup_species("CO2");
    helper.add_lookup_species("H2O");
    _calc = scinew RadPropertyCalculator::RadPropsInterface(); 
#else
    throw InvalidValue("Error: gasRadProps requires that you compile Arches with the RadProps library (try configuring with --enable-wasatch_3p and --with-boost=DIR.)",__FILE__,__LINE__);
#endif
  } else if ( calculator_type == "GauthamWSGG"){
    _calc = scinew RadPropertyCalculator::GauthamWSGG(); 
    helper.add_lookup_species("CO2");
    helper.add_lookup_species("H2O");
    helper.add_lookup_species("mixture_molecular_weight");
  } else { 
    throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 
  } 

  if ( db_calc->findBlock("temperature")){ 
    db_calc->findBlock("temperature")->getAttribute("label", _temperature_name); 
  } else { 
    _temperature_name = "temperature"; 
  }


  bool complete; 
  complete = _calc->problemSetup( db_calc );

  if ( !complete )
    throw InvalidValue("Error: Unable to setup radiation property calculator: "+calculator_type,__FILE__, __LINE__); 


}





void
gasRadProperties::create_local_labels(){
  register_new_variable<CCVariable<double> >(_abskg_name);
}



void
gasRadProperties::register_initialize( VIVec& variable_registry , const bool pack_tasks){
  register_variable( _abskg_name, Uintah::ArchesFieldContainer::COMPUTES, variable_registry );
}

void
gasRadProperties::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& abskg     = tsk_info->get_uintah_field_add<CCVariable<double> >( _abskg_name);

  _calc->initialize_abskg( patch,abskg  ); 

}

void gasRadProperties::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

}

void gasRadProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void gasRadProperties::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

}


void gasRadProperties::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void
gasRadProperties::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

  _calc->setPressure();

  register_variable( _abskg_name , Uintah::ArchesFieldContainer::COMPUTES, variable_registry, time_substep);


    std::vector<std::string> part_sp = _calc->get_sp(); 

    for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
      const VarLabel* label = VarLabel::find(*iter);
      if ( label != 0 ){ 
        register_variable(*iter , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::LATEST,variable_registry, time_substep );
      } else { 
        throw ProblemSetupException("Error: Could not match species with varlabel: "+*iter,__FILE__, __LINE__);
      }
    }
  register_variable(_temperature_name , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::LATEST,variable_registry, time_substep );
  register_variable("volFraction" , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::NEWDW,variable_registry, time_substep );



}


void
gasRadProperties::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
  CCVariable<double>& abskg = *(tsk_info->get_uintah_field<CCVariable<double> >(_abskg_name));
  abskg.initialize(0.0); 

  constCCVariable<double>& temperature = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(_temperature_name));
  constCCVariable<double>& vol_fraction = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("volFraction"));

  std::vector<std::string> part_sp = _calc->get_sp(); 
  std::vector<constCCVariable<double>  > species(0); 
  for ( std::vector<std::string>::iterator iter = part_sp.begin(); iter != part_sp.end(); iter++){
    species.push_back(*(tsk_info->get_const_uintah_field<constCCVariable<double> >(*iter)));
  }

  _calc->compute_abskg( patch, vol_fraction, species, temperature, abskg ); 

  if (_absorption_modifier  > 1.00001 || _absorption_modifier  < 0.99999){ // if the modifier is 1.0, skip this loop
    Uintah::parallel_for( range,  [&](int i, int j, int k){
                 abskg(i,j,k)*=_absorption_modifier ;
               });
  }
}



