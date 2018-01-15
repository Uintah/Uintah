#ifndef Uintah_Component_Arches_spectralProperties_h
#define Uintah_Component_Arches_spectralProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS
//
// /** 
// * @class  SpectralProperties 
// * @author Derek Harris 
// * @date   August 2017 
// * 
// * @brief Computes Spectral properties of CO2 and H2O mixtures. This model is a 4 band model.
// *        The algorithm is valid for temperature from 500k-2400k, with H2O/CO2 molar ratios ranging from .01-4.0,
// *        and optical path lengths ranging from .01 to 60 meters at 1 atm.  
// *        Exceding the bounds of these regions can result in negative absorption weights.  It is particularily bad in composition space.
// *        In this implimentation the input range for composition and temperature is being limited.  [i.e. max(min(X,4),.01)]
// *        Based on Combustion and Flame publication, Bordbar et al. 2014
// *
// * The input file interface for this property should like this in your UPS file: 
// * \code 
// *    <PropertyModelsV2>                                     
// *       <model type="spectralProperties"  label="specteral_abskg" >
// *       </model>
// *    </PropertyModelsV2>
// * \endcode 
// *  
// */ 

namespace Uintah{ 

  class spectralProperties : public TaskInterface{


 public:
    spectralProperties( std::string prop_name, int matl_index  );
    ~spectralProperties();

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info);

    void restart_initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info);

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();


    //Build instructions for this (CO) class.
      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index)
            : _task_name(task_name), _matl_index(matl_index) {}
          ~Builder(){}

          spectralProperties* build()
          { return scinew spectralProperties( _task_name, _matl_index ); }

        private:

          std::string _task_name;
          int _matl_index;

      };

    private: 

const double wecel_C_coeff[5][4][5] {

     {{0.7412956,  -0.9412652,   0.8531866,   -.3342806,    0.0431436 },
      {0.1552073,  0.6755648,  -1.1253940, 0.6040543,  -0.1105453},
      {0.2550242,  -0.605428,  0.8123855,  -0.45322990,  0.0869309},
      {-0.0345199, 0.4112046, -0.5055995, 0.2317509, -0.0375491}},

     {{-0.5244441, 0.2799577 ,   0.0823075,    0.1474987,   -0.0688622},
      {-0.4862117, 1.4092710 ,  -0.5913199,  -0.0553385 , 0.0464663},
      {0.3805403 , 0.3494024 ,  -1.1020090,   0.6784475 , -0.1306996},  
      {0.2656726 , -0.5728350,  0.4579559 ,  -0.1656759 , 0.0229520}},               
      
     {{ 0.582286 , -0.7672319,  0.5289430,   -0.4160689,  0.1109773},
      { 0.3668088, -1.3834490,  0.9085441,  -0.1733014 ,  -0.0016129},
      {-0.4249709, 0.1853509 ,  0.4046178,  -0.3432603 ,  0.0741446},
      {-0.1225365, 0.2924490,  -0.2616436,  0.1052608  ,  -0.0160047}},
      
     {{-.2096994,   0.3204027, -.2468463,   0.1697627,   -0.0420861},
      {-0.1055508,  0.4575210, -0.3334201,  0.0791608,  -0.0035398},
      {0.1429446,  -0.1013694, -0.0811822,  0.0883088,  -0.0202929},
      {0.0300151,  -0.0798076, 0.0764841,  -0.0321935,  0.0050463}},
      
     {{0.0242031 , -.0391017 ,  0.0310940,  -0.0204066,  0.0049188},
      {0.0105857 , -0.0501976,  0.0384236, -0.0098934 ,  0.0006121},   
      {-0.0157408, 0.0130244 ,  0.0062981, -0.0084152 ,  0.0020110},      
      {-0.0028205, 0.0079966 , -0.0079084,  0.003387  , -0.0005364}}};


const double wecel_d_coeff[4][5]
     {{0.0340429, 0.0652305, -0.0463685, 0.0138684, -0.0014450},
      {0.3509457, 0.7465138, -0.5293090, 0.1594423, -0.0166326},
      {4.57074  , 2.1680670, -1.4989010, 0.4917165, -0.0542999},
      {109.81690, -50.923590, 23.432360, -5.163892, 0.4393889}};



      const int _nbands{5};
      const double _C_2{143.88};           ///2nd planck function optical constant (meters*Kelvin)
      double _C_o;                         ///particle optical constant
      std::vector<std::string> _part_sp;
      double _absorption_modifier;
      const VarLabel* _temperature_label; 
      std::string _temperature_name;
      std::string _soot_name;
      bool _LsootOn;
      std::vector<std::string> _abskg_name_vector;
      std::vector<std::string> _abswg_name_vector;

  }; // class spectralRadProperties
}   // namespace Uintah

#endif
