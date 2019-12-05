#ifndef Uintah_Component_Arches_partRadProperties_h
#define Uintah_Component_Arches_partRadProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/MaterialManager.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#ifdef HAVE_RADPROPS
#  include <radprops/Particles.h>
#endif
#include <CCA/Components/Arches/ChemMixV2/ClassicTable.h>

#define DEP_VAR_SIZE 2
#define IND_VAR_SIZE 2

// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS
//
// /** 
// * @class  ADD
// * @author ADD
// * @date   ADD
// * 
// * @brief Computes ADD INFORMATION HERE
// *
// * ADD INPUT FILE INFORMATION HERE: 
// * The input file interface for this property should like this in your UPS file: 
// * \code 
// *   <PropertyModels>
// *     <.......>
// *   </PropertyModels>
// * \endcode 
// *  
// */ 

namespace Uintah{ 

  enum RadPropertiesModel { basic, constantPlanck, constantRossland};

  class RadPropertyCalculator; 

  class partRadProperties : public TaskInterface{


 public:
    partRadProperties( std::string prop_name, int matl_index  );
    ~partRadProperties();

    typedef std::vector<ArchesFieldContainer::VariableInformation> VIVec;

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( VIVec& variable_registry , const bool pack_tasks);

    void register_timestep_init( VIVec& variable_registry , const bool packed_tasks);

    void register_restart_initialize( VIVec& variable_registry , const bool packed_tasks);

    void register_timestep_eval( VIVec& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( VIVec& variable_registry, const int time_substep , const bool packed_tasks){}

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );


    void create_local_labels();

    //Build instructions for this (CO) class.
      class Builder : public TaskInterface::TaskBuilder {

        public:

          Builder( std::string task_name, int matl_index)
            : m_task_name(task_name), m_matl_index(matl_index) {}
          ~Builder(){}

          partRadProperties* build()
          { return scinew partRadProperties( m_task_name, m_matl_index ); }

        private:

          std::string m_task_name;
          int m_matl_index;

      };

      //class Builder
        //: public TaskInterface::TaskBuilder { 

          //public: 

            //Builder( std::string name, MaterialManagerP& materialManager,ArchesLabel * fieldLabels) : _name(name), _materialManager(materialManager),_fieldLabels(fieldLabels) {};
            //~Builder(){}; 

            //partRadProperties* build()
            //{ return scinew partRadProperties( _name, _materialManager, _fieldLabels ); };

          //private: 

            //std::string _name; 
            //MaterialManagerP& _materialManager; 
            //ArchesLabel* _fieldLabels;

        //}; // class Builder 


    private: 
      
      RadPropertyCalculator::PropertyCalculatorBase* _calc; 
      std::string _particle_calculator_type;
      const VarLabel* _temperature_label; 
      
      std::string _temperature_name;
      std::vector< std::string > _abskp_name_vector;
      std::vector< std::string > _complexIndexReal_name;   // particle absorption coefficient
      std::string _abskp_name;
      std::string _scatkt_name        {"scatkt"}; 
      std::string _asymmetryParam_name{"asymmetryParam"};

      int _nQn_part ;                                // number of quadrature nodes in DQMOM
      std::vector<std::string>  _temperature_name_v;          // DQMOM Temperature name
      std::vector<std::string>  _size_name_v;                 // DQMOM size_name
      std::vector<std::string>  _weight_name_v;          ///> name of DQMOM weights
      std::vector<std::string>  _RC_name_v;              ///> name of Raw Coal variable
      std::vector<std::string>  _Char_name_v;            ///> name of char coal 
      std::vector<double>  _ash_mass_v;                  ///> particle ash mass (constant)

      bool  _isCoal ;
      bool  _scatteringOn ;


      std::vector < std::string > _composition_names;

#ifdef HAVE_RADPROPS
      RadProps::ParticleRadCoeffs* _part_radprops;
      RadProps::ParticleRadCoeffs3D* _3Dpart_radprops;
      std::complex<double> _HighComplex;
      std::complex<double> _LowComplex;
#endif


      bool _p_planck_abskp; 
      bool _p_ros_abskp; 

      double _constAsymmFact;
      double _Qabs;

      Interp_class<DEP_VAR_SIZE>*  myTable;

       // coal optics data members
      double _rawCoalReal;
      double _rawCoalImag;
      double _charReal;
      double _charImag;
      double _ashReal;
      double _ashImag;
      int _ncomp;
      
     int _nIVs;  /// number of independent variables for table lookup
     int _nDVs{0};  /// number of dependent variables for table lookup

      double _absorption_modifier;
      double _scattering_modifier;
      double  _charAsymm;
      double  _rawCoalAsymm;
      double  _ashAsymm;



  }; // class partRadProperties
}   // namespace Uintah


#endif
