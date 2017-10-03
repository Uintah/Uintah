#ifndef Uintah_Component_Arches_partRadProperties_h
#define Uintah_Component_Arches_partRadProperties_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
#ifdef HAVE_RADPROPS
#  include <radprops/Particles.h>
#endif

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

          partRadProperties* build()
          { return scinew partRadProperties( _task_name, _matl_index ); }

        private:

          std::string _task_name;
          int _matl_index;

      };

      //class Builder
        //: public TaskInterface::TaskBuilder { 

          //public: 

            //Builder( std::string name, SimulationStateP& shared_state,ArchesLabel * fieldLabels) : _name(name), _shared_state(shared_state),_fieldLabels(fieldLabels) {};
            //~Builder(){}; 

            //partRadProperties* build()
            //{ return scinew partRadProperties( _name, _shared_state, _fieldLabels ); };

          //private: 

            //std::string _name; 
            //SimulationStateP& _shared_state; 
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

      int _nQn_part ;                                // number of quadrature nodes in DQMOM
      std::vector<std::string>  _temperature_name_v;          // DQMOM Temperature name
      std::vector<std::string>  _size_name_v;                 // DQMOM size_name
      std::vector<std::string>  _weight_name_v;          ///> name of DQMOM weights
      std::vector<std::string>  _RC_name_v;              ///> name of Raw Coal variable
      std::vector<std::string>  _Char_name_v;            ///> name of char coal 
      std::vector<double>  _ash_mass_v;                  ///> particle ash mass (constant)

      bool  _isCoal ;
      bool  _scatteringOn ;
      std::string _scatkt_name; 

      std::string _asymmetryParam_name;

      std::vector < std::string > _composition_names;

      RadProps::ParticleRadCoeffs* _part_radprops;
      RadProps::ParticleRadCoeffs3D* _3Dpart_radprops;

      std::complex<double> _HighComplex;
      std::complex<double> _LowComplex;

      bool _p_planck_abskp; 
      bool _p_ros_abskp; 

      double _constAsymmFact;
      double _Qabs;


           // coal optics data members
          double _rawCoalReal;
          double _rawCoalImag;
          double _charReal;
          double _charImag;
          double _ashReal;
          double _ashImag;
          std::complex<double> _complexLo;  
          std::complex<double> _complexHi;  
          int _ncomp;
          

          double _absorption_modifier;
          double  _charAsymm;
          double  _rawCoalAsymm;
          double  _ashAsymm;



  }; // class partRadProperties
}   // namespace Uintah


#endif
