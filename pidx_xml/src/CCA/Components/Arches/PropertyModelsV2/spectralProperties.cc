#include <CCA/Components/Arches/PropertyModelsV2/spectralProperties.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <Core/Containers/StaticArray.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
////Method: Constructor
////---------------------------------------------------------------------------
spectralProperties::spectralProperties( std::string task_name, int matl_index ) : TaskInterface( task_name, matl_index)
{
}
  
  
//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
spectralProperties::~spectralProperties( )
{
}

//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void spectralProperties::problemSetup(  Uintah::ProblemSpecP& db )
{

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species("CO2");
  helper.add_lookup_species("H2O");
  helper.add_lookup_species("mixture_molecular_weight");
  _part_sp.push_back("CO2"); // must be in order of CO2 -> H2O -> mixture_molec_weight
  _part_sp.push_back("H2O");
  _part_sp.push_back("mixture_molecular_weight"); 

  db->getWithDefault("absorption_modifier",_absorption_modifier,1.0);
  _temperature_name = "temperature"; 

  std::string _abskg_name_base="abskg";
  std::string _weight_name_base="abswg";
  _abskg_name_vector = std::vector<std::string> (4);
  _abswg_name_vector = std::vector<std::string> (4);

  for (int i=0; i< _nbands ; i++){
    std::stringstream out1;
    out1 << _abskg_name_base << "_" << i;
    _abskg_name_vector[i] = out1.str();

    std::stringstream out2;
    out2 << _weight_name_base << "_" << i;
    _abswg_name_vector[i] = out2.str();

  }
}


void
spectralProperties::create_local_labels(){

  for (int i=0; i< _nbands ; i++){
    register_new_variable<CCVariable<double> >(_abskg_name_vector[i]);
    register_new_variable<CCVariable<double> >(_abswg_name_vector[i]);
  }

}


void
spectralProperties::register_initialize( VIVec& variable_registry , const bool pack_tasks){
  for (int i=0; i< _nbands ; i++){
    register_variable( _abskg_name_vector[i], Uintah::ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( _abswg_name_vector[i], Uintah::ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

void
spectralProperties::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  for (int i=0; i< _nbands ; i++){
  CCVariable<double>& abskg     = tsk_info->get_uintah_field_add<CCVariable<double> >( _abskg_name_vector[i]);
  CCVariable<double>& abswg     = tsk_info->get_uintah_field_add<CCVariable<double> >( _abswg_name_vector[i]);
  abskg.initialize(0.0);
  abswg.initialize(0.0);
  }
}

void spectralProperties::register_restart_initialize( VIVec& variable_registry , const bool packed_tasks){

}

void spectralProperties::restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void spectralProperties::register_timestep_init( VIVec& variable_registry , const bool packed_tasks){

}


void spectralProperties::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

}

void
spectralProperties::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){


  for (int i=0; i< _nbands ; i++){
    register_variable( _abskg_name_vector[i] , Uintah::ArchesFieldContainer::COMPUTES, variable_registry, time_substep);
    register_variable( _abswg_name_vector[i] , Uintah::ArchesFieldContainer::COMPUTES, variable_registry, time_substep);
  }



    for ( std::vector<std::string>::iterator iter = _part_sp.begin(); iter != _part_sp.end(); iter++){
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
spectralProperties::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  StaticArray< CCVariable<double> > abskg(_nbands); 
  StaticArray< CCVariable<double> > abswg(_nbands); 
  for (int i=0; i< _nbands ; i++){

    tsk_info->get_unmanaged_uintah_field<CCVariable<double> >(_abskg_name_vector[i],abskg[i]);
    tsk_info->get_unmanaged_uintah_field<CCVariable<double> >(_abswg_name_vector[i],abswg[i]);

    abskg[i].initialize(0.0); 
    abswg[i].initialize(0.0); 
  }

  constCCVariable<double>& temperature = *(tsk_info->get_const_uintah_field<constCCVariable<double> >(_temperature_name));
  constCCVariable<double>& vol_fraction = *(tsk_info->get_const_uintah_field<constCCVariable<double> >("volFraction"));

  std::vector<constCCVariable<double>  > species(0); 
  for ( std::vector<std::string>::iterator iter = _part_sp.begin(); iter != _part_sp.end(); iter++){
    species.push_back(*(tsk_info->get_const_uintah_field<constCCVariable<double> >(*iter)));
  }

  const int n_coeff=5;
  const int T_normalize=1200;  // 1200k per Bordbar et al. 2014
  Uintah::parallel_for( range,  [&](int i, int j, int k){
              if (vol_fraction(i,j,k)> 0.5){


                 /// absorption coefficients and weights computed from Bordbar et al. 2014
                 const double CO2= species[0](i,j,k) / species[2](i,j,k) /44.01; // species[2] is 1/MW_mixture
                 const double H2O= species[1](i,j,k) / species[2](i,j,k) /18.02;
                                 
                 const double m = std::max(std::min(H2O/CO2,4.0),0.01); // prevent extrapolation from data fit
                 const double T_r = temperature(i,j,k)/T_normalize; 

                 std::vector<std::vector<double> > b_vec(_nbands,std::vector<double>(n_coeff,0.0));

                 double m_k=1.0; // m^k
                 for (int kk=0; kk < n_coeff ;  kk++){
                   for (int jj=0; jj < _nbands;  jj++){
                     for (int ii=0; ii < n_coeff;  ii++){
                      b_vec[jj][ii]+=wecel_C_coeff[kk][jj][ii]*m_k;
                     }
                   }
                 m_k*=m; 
                 }

                 double T_r_k=1.0; //T_r^k
                 m_k=1.0;
                 for (int kk=0; kk < n_coeff;  kk++){
                   for (int ii=0; ii< _nbands ; ii++){
                     
                     abswg[ii](i,j,k)+=b_vec[ii][kk]*T_r_k;
                     abskg[ii](i,j,k)+=wecel_d_coeff[ii][kk]*m_k*(H2O+CO2); // table was built assuming H2O + CO2 = 1.0
                   }
                   T_r_k*=T_r; 
                   m_k*=m; 
                 }
              }
   });

   if (_absorption_modifier  > 1.00001 || _absorption_modifier  < 0.99999){ // if the modifier is 1.0, skip this loop
      Uintah::parallel_for( range,  [&](int i, int j, int k){
                 for (int ix=0; ix< _nbands ; ix++){
                   abskg[ix](i,j,k)*=_absorption_modifier ;
                 }
      });
   }
}



