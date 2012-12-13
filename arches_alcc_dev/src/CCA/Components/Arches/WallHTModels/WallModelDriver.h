#ifndef Uintah_Components_Arches_WallModelDriver_h
#define Uintah_Components_Arches_WallModelDriver_h

#include <Core/Grid/LevelP.h>
#include <Core/Util/DebugStream.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>


//============================================

/** 
 * @class  Wall Model driver 
 * @author Jeremy Thornock
 * @date   Oct, 2012
 *
 * @brief  Driver class for various wall models. 
 *
 */

namespace Uintah{ 

  // setenv SCI_DEBUG WALL_MODEL_DRIVER_DEBUG:+   (tcsh)
  // export SCI_DEBUG="WALL_MODEL_DRIVER_DEBUG:+" (bash)
  static DebugStream cout_wmd_dbg("WALL_MODEL_DRIVER",false);

  class VarLabel; 

  class WallModelDriver { 

    public: 

      WallModelDriver( SimulationStateP& shared_state ); 
      ~WallModelDriver(); 

      /** @brief Input file interface **/
      void problemSetup( const ProblemSpecP& db ); 

      /** @brief Compute the heat tranfer to the walls/tubes -- does all to all **/
      void sched_doWallHT_alltoall( const LevelP& level, SchedulerP& sched, const int time_substep );

      /** @brief Compute the heat tranfer to the walls/tubes **/
      void sched_doWallHT( const LevelP& level, SchedulerP& sched, const int time_substep );

      struct HTVariables {

        CCVariable<double> T; 
        constCCVariable<int> celltype; 
        constCCVariable<double > hf_e; 
        constCCVariable<double > hf_w; 
        constCCVariable<double > hf_n; 
        constCCVariable<double > hf_s; 
        constCCVariable<double > hf_t; 
        constCCVariable<double > hf_b; 

      };

    private: 


      /** @brief The base class definition for all derived wall heat transfer models **/ 
      class HTModelBase{

        public: 

          HTModelBase(){}; 
          virtual ~HTModelBase(){}; 

          virtual void problemSetup( const ProblemSpecP& input_db ) = 0;
          virtual void computeHT( const Patch* patch, HTVariables* vars ) = 0; 

        private: 

          std::string _model_name; 

      };

      /** @brief A simple wall heat transfer model for domain walls only **/
      class SimpleHT : public HTModelBase { 

        public: 

          SimpleHT(); 
          ~SimpleHT(); 

          void problemSetup( const ProblemSpecP& input_db ); 
          void computeHT( const Patch* patch, HTVariables* vars ); 

        private: 

          double _k;         ///< Thermal conductivity 
          double _dy;        ///< Wall thickness 
          double _T_inner;   ///< Inner wall temperature

      };

      std::string _T_label_name; 

      //varlabel references
      const VarLabel* _T_label;
      const VarLabel* _cellType_label; 
      const VarLabel* _HF_E_label; 
      const VarLabel* _HF_W_label; 
      const VarLabel* _HF_N_label; 
      const VarLabel* _HF_S_label; 
      const VarLabel* _HF_T_label; 
      const VarLabel* _HF_B_label; 

      SimulationStateP& _shared_state; 

      std::vector<HTModelBase*> _all_ht_models; 

      int _matl_index; 

      void doWallHT( const ProcessorGroup* my_world,
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw, 
                     const int time_substep );

      void doWallHT_alltoall( const ProcessorGroup* my_world,
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw, 
                              const int time_substep );

      bool check_varlabels(){ 
        bool result = true; 
        std::string offender = "none";

        if ( _T_label == 0 ){ 
          result = false; 
          offender = "temperature";
        } 
        if ( _cellType_label == 0 ){ 
          result = false; 
          offender = "cell_type"; 
        } 
        if ( _HF_E_label == 0 ){ 
          result = false; 
          offender = "heat_flux_e"; 
        } 
        if ( _HF_W_label == 0 ){ 
          result = false; 
          offender = "heat_flux_w"; 
        } 
        if ( _HF_N_label == 0 ){ 
          result = false; 
          offender = "heat_flux_n"; 
        } 
        if ( _HF_S_label == 0 ){ 
          result = false; 
          offender = "heat_flux_s"; 
        } 
        if ( _HF_T_label == 0 ){ 
          result = false; 
          offender = "heat_flux_t"; 
        } 
        if ( _HF_B_label == 0 ){ 
          result = false; 
          offender = "heat_flux_b"; 
        } 

        cout_wmd_dbg << " WallModelDriver:: The missing varlabel = " << offender << std::endl;

        return result; 
        
      } 


  }; 
} // namespace Uintah

#endif
