#ifndef Uintah_Components_Arches_WallModelDriver_h
#define Uintah_Components_Arches_WallModelDriver_h

#include <Core/Grid/LevelP.h>
#include <Core/Util/DebugStream.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>


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
      //void sched_doWallHT_alltoall( const LevelP& level, SchedulerP& sched, const int time_substep );

      /** @brief Compute the heat tranfer to the walls/tubes **/
      void sched_doWallHT( const LevelP& level, SchedulerP& sched, const int time_substep );

      struct HTVariables {

        CCVariable<double> T; 
        CCVariable<double> T_copy; 
        constCCVariable<double> T_old;
        constCCVariable<int> celltype; 
        constCCVariable<double > incident_hf_e; 
        constCCVariable<double > incident_hf_w; 
        constCCVariable<double > incident_hf_n; 
        constCCVariable<double > incident_hf_s; 
        constCCVariable<double > incident_hf_t; 
        constCCVariable<double > incident_hf_b; 
        constCCVariable<Vector > cc_vel; 

      };

    private: 

      int _calc_freq;                    ///< Wall heat transfer model calculation frequency
      std::string _T_label_name;         ///< Temperature label name
      SimulationStateP& _shared_state; 
      int _matl_index; 

      // Net heat flux var labels: 
      const VarLabel* _T_copy_label; 

      //varlabel references to other variables
      const VarLabel* _T_label;
      const VarLabel* _cc_vel_label; 
      const VarLabel* _cellType_label; 
      const VarLabel* _HF_E_label; 
      const VarLabel* _HF_W_label; 
      const VarLabel* _HF_N_label; 
      const VarLabel* _HF_S_label; 
      const VarLabel* _HF_T_label; 
      const VarLabel* _HF_B_label; 

      void doWallHT( const ProcessorGroup* my_world,
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw, 
                     const int time_substep );

      //void doWallHT_alltoall( const ProcessorGroup* my_world,
      //                        const PatchSubset* patches, 
      //                        const MaterialSubset* matls, 
      //                        DataWarehouse* old_dw, 
      //                        DataWarehouse* new_dw, 
      //                        const int time_substep );

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
        if ( _cc_vel_label == 0 ){
          result = false; 
          offender = "CCVelocity";
        }

        cout_wmd_dbg << " WallModelDriver:: The missing varlabel = " << offender << std::endl;

        return result; 
        
      } 

      //-----------------------------------------------------------------
      //
      // Derived models 
      //
      // ----------------------------------------------------------------


      // -----------------------
      // Simple HT
      // -----------------------

      /** @brief The base class definition for all derived wall heat transfer models **/ 
      class HTModelBase{

        public: 

          HTModelBase(){}; 
          virtual ~HTModelBase(){}; 

          virtual void problemSetup( const ProblemSpecP& input_db ) = 0;
          virtual void computeHT( const Patch* patch, HTVariables& vars ) = 0; 
          virtual void copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_old, constCCVariable<int>& cell_type ) = 0; 

        private: 

          std::string _model_name; 

        protected: 

          /** @brief Test if the (i,j,k) is inside or outside of the geometry **/ 
          inline bool in_or_out( IntVector c, GeometryPieceP piece, const Patch* patch ){ 

            bool test = false; 

            Point p = patch->cellPosition( c ); 
            if ( piece->inside( p ) ) { 
              test = true; 
            } 

            return test; 

          }; 

      };

      std::vector<HTModelBase*> _all_ht_models; 


      /** @brief A simple wall heat transfer model for domain walls only **/
      class SimpleHT : public HTModelBase { 

        public: 

          SimpleHT(); 
          ~SimpleHT(); 

          void problemSetup( const ProblemSpecP& input_db ); 
          void computeHT( const Patch* patch, HTVariables& vars ); 
          void copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_copy, constCCVariable<int>& cell_type ); 

        private: 

          double _k;         ///< Thermal conductivity 
          double _dy;        ///< Wall thickness 
          double _T_inner;   ///< Inner wall temperature
          const double _sigma_constant;      ///< Stefan Boltzman constant [W/(m^2 K^4)]
          double _T_max;     ///< Maximum allowed wall temperature
          double _T_min;     ///< Minimum allowed wall temperature
          double _relax;     ///< A relaxation coefficient to help stability (eg, wall temperature changes too fast)...but not necessarily with accuracy

      };

      // -----------------------
      // Region HT
      // -----------------------

      /** @brief A simple wall heat transfer model for domain walls only **/
      class RegionHT : public HTModelBase { 

        public: 

          RegionHT(); 
          ~RegionHT(); 

          void problemSetup( const ProblemSpecP& input_db ); 
          void computeHT( const Patch* patch, HTVariables& vars ); 
          void copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_copy, constCCVariable<int>& cell_type ); 

        private: 


          const double _sigma_constant;      ///< Stefan Boltzman constant [W/(m^2 K^4)]
          double _init_tol;                  ///< initial tolerance for the iterative solver
          double _tol;                       ///< solver tolerance 
          int _max_it;                       ///< maximum iterations allowed 

          struct WallInfo { 
              double k; 
              double dy; 
              double T_inner; 
              double relax;     ///< A relaxation coefficient to help stability (eg, wall temperature changes too fast)...but not necessarily with accuracy
              double max_TW;     ///< maximum wall temperature
              double min_TW;     ///< minimum wall temperature
            std::vector<GeometryPieceP> geometry; 
          };

          std::vector<WallInfo> _regions; 

          std::vector<IntVector> _d; 

          inline constCCVariable<double> get_flux( int i, HTVariables& vars ){ 

            constCCVariable<double> q; 
            switch (i) {
              case 0:
                q = vars.incident_hf_w;
                break; 
              case 1:
                q = vars.incident_hf_e;
                break; 
              case 2:
                q = vars.incident_hf_s;
                break; 
              case 3:
                q = vars.incident_hf_n;
                break; 
              case 4:
                q = vars.incident_hf_b;
                break; 
              case 5:
                q = vars.incident_hf_t;
                break; 
              default: 
                break; 
            }
            return q; 

          };

      };



  }; 




} // namespace Uintah

#endif
