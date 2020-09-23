#include <CCA/Components/Arches/PropertyModelsV2/sumRadiation.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <Core/Util/DOUT.hpp>
#include <ostream>
#include <cmath>

namespace Uintah{

Dout dbg_sumRad("Arches_sumRad", "Arches::PropertyModelsV2::sumRadiation", "outputs what abskt is comprised of", false);

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace sumRadiation::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace sumRadiation::loadTaskInitializeFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &sumRadiation::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &sumRadiation::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &sumRadiation::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace sumRadiation::loadTaskEvalFunctionPointers()
{
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &sumRadiation::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &sumRadiation::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &sumRadiation::eval<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace sumRadiation::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace sumRadiation::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
void
sumRadiation::problemSetup( ProblemSpecP& db ){

  ProblemSpecP db_prop = db;

  //bool foundPart=false;  // intended to be used in the future
  int igasPhase=0;
  for ( ProblemSpecP db_model = db_prop->findBlock("model"); db_model != nullptr; db_model=db_model->findNextBlock("model") ){

    std::string type;
    db_model->getAttribute("type", type);

    if ( type == "gasRadProperties" ){

      igasPhase++;
      std::string fieldName;
      db_model->getAttribute("label",fieldName);
      m_absk_names.push_back(fieldName);
    }
    else if ( type == "partRadProperties" ) {

      std::string fieldName;
      db_model->getAttribute("label",fieldName);
      m_absk_names.push_back(fieldName);
      //foundPart=true;
    }
    else if ( type == "spectralProperties" ){

      igasPhase++;
      std::string soot_name;
      db_model->get("sootVolumeFrac",soot_name);

      if (soot_name==""){
        proc0cout << " WARNING:: NO SOOT FOUND FOR RADIATION  \n";
      }else{
        m_absk_names.push_back("absksoot"); // only needed for spectral radiation because of grey soot and colorful gas
      }
    }

    if (igasPhase > 1){
      throw ProblemSetupException("Multiple gas phase radiation property models found! Arches doesn't know which one it should use.",__FILE__, __LINE__);
    }
  }

  if (igasPhase<1){ // for tabulated gas properties
    ChemHelper& helper = ChemHelper::self();
    helper.add_lookup_species("abskg");
  }

//----------------------set name of total absorption coefficient ------------//

  ProblemSpecP db_source = db_prop->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources") ;

  for ( ProblemSpecP db_src = db_source->findBlock("src"); db_src != nullptr; db_src = db_src->findNextBlock("src")){

    std::string radiation_model;
    db_src->getAttribute("type", radiation_model);

    if (radiation_model == "do_radiation" ){

      std::string my_abskt_name = "notSet";
      ProblemSpecP db_abskt = db_src->findBlock("abskt");

      if ( db_abskt ){
        db_abskt->getAttribute("label", my_abskt_name);
      }
      else{
        throw ProblemSetupException("Absorption coefficient not specified.",__FILE__, __LINE__);
      }

      if (m_abskt_name != "undefined" && my_abskt_name != m_abskt_name ){
        proc0cout << "WARNING: Multiple Radiation solvers detected, but they are using different absorption coefficients. \n";
      }
      m_abskt_name = my_abskt_name;


      //--------Now check if scattering is on for DO----//
      bool scatteringOn=false;

      db_src->findBlock("DORadiationModel")->getWithDefault("ScatteringOn",scatteringOn,false) ;
      if (scatteringOn){
        m_absk_names.push_back("scatkt");
      }
    }
  }
  
  //__________________________________
  //  output the variables that abskt is comprised
  proc0cout << std::right<< std::setw(20) << m_abskt_name << " = ";

  size_t n = m_absk_names.size();

  for (size_t i=0; i<n; i++){
    std::string c ( (i+1 == n) ? "\n" : " + " );    // "+" or "\n"
    proc0cout << m_absk_names[i] << c;
  }
}


//--------------------------------------------------------------------------------------------------
void
sumRadiation::create_local_labels(){
  register_new_variable<CCVariable<double> >(m_abskt_name);
}

//--------------------------------------------------------------------------------------------------
void
sumRadiation::register_initialize( VIVec& variable_registry ,
                                   const bool pack_tasks){

  register_variable( m_abskt_name,  ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable("volFraction" , ArchesFieldContainer::REQUIRES,0,ArchesFieldContainer::NEWDW,variable_registry);

  for (unsigned int i=0; i<m_absk_names.size(); i++){
    register_variable(m_absk_names[i] , Uintah::ArchesFieldContainer::REQUIRES, variable_registry);
  }
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void
sumRadiation::initialize( const Patch* patch,
                          ArchesTaskInfoManager* tsk_info,
                          ExecutionObject<ExecSpace, MemSpace>& execObj ){

  auto abskt = tsk_info->get_field<CCVariable<double>, double, MemSpace>(m_abskt_name);
  auto volFrac = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>("volFraction");

  Uintah::parallel_initialize(execObj, 1.0, abskt);

  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex());

  const double abs_frac = 1./(double)m_absk_names.size();

  for (unsigned int i=0; i<m_absk_names.size(); i++){

    auto abskf = tsk_info->get_field<constCCVariable<double>, const double, MemSpace>(m_absk_names[i]);

    Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA( int i, int j, int k ){

      if (volFrac(i,j,k) > 1e-16){
        abskt(i,j,k) = abskt(i,j,k) + abskf(i,j,k) - abs_frac;          // Dimensionally this is inconsistent.  --Todd
      } else {
        abskt(i,j,k) = 1.0;
      }

    });

    if (m_absk_names.size()==0){
      Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA( int i, int j, int k ){
        abskt(i,j,k)=(volFrac(i,j,k) > 1e-16) ? 0.0  : 1.0;
      });
    }

  }  // loop over names
}

//--------------------------------------------------------------------------------------------------
void sumRadiation::register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks){
  register_initialize( variable_registry , false);
}

//--------------------------------------------------------------------------------------------------
template <typename ExecSpace, typename MemSpace>
void sumRadiation::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  initialize( patch, tsk_info, execObj );
}

} //namespace Uintah
