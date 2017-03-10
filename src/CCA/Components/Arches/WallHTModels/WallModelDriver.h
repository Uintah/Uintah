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

      enum RAD_MODEL_TYPE { DORADIATION, RMCRT };

      WallModelDriver( SimulationStateP& shared_state );
      ~WallModelDriver();

      /** @brief Input file interface **/
      void problemSetup( const ProblemSpecP& db );

      /** @brief Compute the heat tranfer to the walls/tubes -- does all to all **/
      //void sched_doWallHT_alltoall( const LevelP& level, SchedulerP& sched, const int time_substep );

      /** @brief Compute the heat tranfer to the walls/tubes **/
      void sched_doWallHT( const LevelP& level, SchedulerP& sched, const int time_substep );

      /** @brief Copy the real T wall (only for wall cells) into the temperature field AFTER table lookup. **/
      void sched_copyWallTintoT( const LevelP& level, SchedulerP& sched );

      struct HTVariables {

        double time;
        int em_model_type;
        double delta_t;
        CCVariable<double> T;
        CCVariable<double> T_copy;
        CCVariable<double> T_real;
        CCVariable<double> deposit_thickness;
        CCVariable<double> emissivity;
        CCVariable<double> thermal_cond_en;
        CCVariable<double> thermal_cond_sb;
        CCVariable<double> deposit_velocity;
        constCCVariable<double> deposit_velocity_old;
        constCCVariable<double> ave_deposit_velocity;
        constCCVariable<double> d_vol_ave;
        constCCVariable<double> T_real_old;
        constCCVariable<double> T_old;
        constCCVariable<int> celltype;
        constCCVariable<double> incident_hf_e;
        constCCVariable<double> incident_hf_w;
        constCCVariable<double> incident_hf_n;
        constCCVariable<double> incident_hf_s;
        constCCVariable<double> incident_hf_t;
        constCCVariable<double> incident_hf_b;
        constCCVariable<double> deposit_thickness_old;
        constCCVariable<double> emissivity_old;
        constCCVariable<double> thermal_cond_en_old;
        constCCVariable<double> thermal_cond_sb_old;
        CCVariable<Stencil7> total_hf;
        constCCVariable<Vector > cc_vel;
        WallModelDriver::RAD_MODEL_TYPE model_type;

      };

    private:
      std::string _dep_vel_name;
      bool do_coal_region;
      int _calc_freq;                    ///< Wall heat transfer model calculation frequency
      std::string _T_label_name;         ///< Temperature label name
      SimulationStateP& _shared_state;
      int _matl_index;                   ///< Material index

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
      const VarLabel* _Total_HF_label;
      const VarLabel* _True_T_Label;
      const VarLabel* _ave_dep_vel_label;
      const VarLabel* _deposit_velocity_label;
      const VarLabel* _deposit_thickness_label;
      const VarLabel* _d_vol_ave_label;
      const VarLabel* _emissivity_label;
      const VarLabel* _thermal_cond_en_label;
      const VarLabel* _thermal_cond_sb_label;

      void doWallHT( const ProcessorGroup* my_world,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     const int time_substep );

      void copyWallTintoT( const ProcessorGroup* my_world,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw );

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
        if ( _cc_vel_label == 0 ){
          result = false;
          offender = "CCVelocity";
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

      inline static std::string get_dep_vel_name( const ProblemSpecP& input_db ){
        ProblemSpecP db = input_db;
        std::string new_name;
        db->require( "deposit_velocity_name", new_name );
        return new_name;
      }
      
      inline static int get_emissivity_model_type( const ProblemSpecP& input_db ){
        ProblemSpecP db = input_db;
        std::string model_type;
        int model_int;
        db->getWithDefault( "emissivity_model_type", model_type,"constant");
        if (model_type=="constant"){
          model_int = 1;
        } else if ( model_type=="dynamic"){
          model_int = 2;
        } else {
          throw InvalidValue("Error: emissivity_model_type must be either constant or dynamic.", __FILE__, __LINE__);
        }
          return model_int;
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
          virtual void computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T ) = 0;
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
        


          struct EmissivityBase {
            virtual void model(double &e, const double &C, double &T, double &Dp, double &tau)=0;
            virtual ~EmissivityBase(){}};
          
          EmissivityBase* m_em_model; 

          struct constant_e : EmissivityBase {
            void model(double &e, const double &C, double &T, double &Dp, double &tau) {
               e = C;
            }
              ~constant_e(){}
            };
            
            struct dynamic_e : EmissivityBase {
              dynamic_e(ProblemSpecP db_model){
                std::string ash_type;
                db_model->getWithDefault( "coal_name", ash_type, "generic_coal");
                db_model->getWithDefault( "frenkel_constant", A_frenkel, 1.225);
                if (ash_type == "indonesian"){
                  a_sv = -1.503e4;
                  b_sv = -1.031;
                  c_sv = 5.366;
                  a_agg = 0.003329;
                  b_agg = 8.575;
                  c_agg = 0.3315;
                  dp_eff_max=0.674746603591938;
                  dp_eff_min=0.333333333333333;
                  dpmax = 3000*1e-6;
                  coeff_num = {0.133872433468528, -0.085588305614014, 0.420224738232270, 0.345964536984323, 0.157355184642739, -0.420810405519288};
                  coeff_den = {1.000000000000000, 0.031154322452954, 0.261038846958539, -0.019838837050095, 0.033559752459297, -0.641137462770430};
                  xscale = {750, 0.001496250000000}; 
                  xcenter = {1050, 0.00150375}; 
                  yscale = 0.082268841760067; 
                  ycenter = 0.940464141548903; 
                  fresnel={0.9671, -1.076e-06, -0.1613, -0.005533};
                } else {
                  throw InvalidValue("Error, coal_name wasn't recognized in dynamic ash emissivity data-base. ", __FILE__, __LINE__);
                }

            }
              double a_sv;
              double b_sv;
              double c_sv;
              double A_frenkel;
              double a_agg;
              double b_agg;
              double c_agg;
              double dp_eff_max;
              double dp_eff_min;
              double dpmax;
              std::vector<double> coeff_num;
              std::vector<double> coeff_den;
              std::vector<double> xscale; 
              std::vector<double> xcenter; 
              double yscale; 
              double ycenter; 
              std::vector<double> fresnel;
              void model(double &e, const double &C, double &T, double &Dp, double &tau) {
                
                // surface tension and viscosity model:
                // power law fit: log10(st/visc) = a*T^b+c
                double log10SurfT_div_Visc = a_sv*std::pow(T,b_sv)+c_sv; // [=] log10(m-s)
                double SurfT_div_Visc = std::pow(10,log10SurfT_div_Visc); // [=] m-s
                 
                // Frenkel's model for sintering
                double x_r = A_frenkel*sqrt(SurfT_div_Visc*tau/Dp/2.0); // Frenkel's model for sintering [=] m/m
                
                // agglomeration model related x/r to effective particle size
                // power law fit: dp_eff_scaled = a*(x/r+1)^b+c
                double rvec=std::min(a_agg*std::pow((x_r+1),b_agg)+c_agg,dp_eff_max);
                double m_dp = (dpmax-Dp)/(dp_eff_max-dp_eff_min);
                double b_dp = Dp - m_dp*dp_eff_min;
                double d_eff=m_dp*rvec+b_dp;
          
                // mie emissivity as a function of temperature and effective particle size
                // rational quardratice fit: y = (a1+a2*x1+a3*x2+a4*x1^2+a5*x1*x2+a6*x2^2)/(b1+b2*x1+b3*x2+b4*x1^2+b5*x1*x2+b6*x2^2) 
                double T_sc = (T-xcenter[0])/xscale[0];
                double d_eff_sc = (d_eff-xcenter[1])/xscale[1];
                double xv[6]={1, T_sc, d_eff_sc, std::pow(T_sc,2), T_sc*d_eff_sc, std::pow(d_eff_sc,2.0)};
                double num=0;
                double den=0;
                for ( int I=0; I < 6; I++ ) {
                  num+=coeff_num[I]*xv[I];
                  den+=coeff_den[I]*xv[I];
                }
                e = num/den;
                e = e*yscale + ycenter;
                // finally set emissivity to fresnel emissivity (slagging limit) if mie-theory emissivity is too high.
                // 2nd order exponential fit: ef = a*exp(b*T)+c*exp(d*T);
                double ef=fresnel[0]*std::exp(fresnel[1]*T) + fresnel[2]*std::exp(fresnel[3]*T); 
                e=std::min(ef,e);
              }
              ~dynamic_e(){}
            };


      };

      std::vector<HTModelBase*> _all_ht_models;


      /** @brief A simple wall heat transfer model for domain walls only **/
      class SimpleHT : public HTModelBase {

        public:

          SimpleHT();
          ~SimpleHT();

          void problemSetup( const ProblemSpecP& input_db );
          void computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T );
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
          void computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T );
          void copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_copy, constCCVariable<int>& cell_type );

        private:


          const double _sigma_constant;      ///< Stefan Boltzman constant [W/(m^2 K^4)]
          double _init_tol;                  ///< initial tolerance for the iterative solver
          double _tol;                       ///< solver tolerance
          int _max_it;                       ///< maximum iterations allowed

          struct WallInfo {
              double k;
              double dy;
              double emissivity;
              double T_inner;
              double relax;     ///< A relaxation coefficient to help stability (eg, wall temperature changes too fast)...but not necessarily with accuracy
              double max_TW;     ///< maximum wall temperature
              double min_TW;     ///< minimum wall temperature
              std::vector<GeometryPieceP> geometry;
          };

          std::vector<WallInfo> _regions;

          std::vector<IntVector> _d;

          inline constCCVariable<double>* get_flux( int i, HTVariables& vars ){

            constCCVariable<double>* q = NULL;
            switch (i) {
              case 0:
                q = &(vars.incident_hf_w);
                break;
              case 1:
                q = &(vars.incident_hf_e);
                break;
              case 2:
                q = &(vars.incident_hf_s);
                break;
              case 3:
                q = &(vars.incident_hf_n);
                break;
              case 4:
                q = &(vars.incident_hf_b);
                break;
              case 5:
                q = &(vars.incident_hf_t);
                break;
              default:
                break;
            }
            return q;

          };
      };

      // -----------------------
      // CoalRegion HT
      // -----------------------

      /** @brief A simple wall heat transfer model for domain walls only **/
      class CoalRegionHT : public HTModelBase {

        public:

          CoalRegionHT();
          ~CoalRegionHT();

          void problemSetup( const ProblemSpecP& input_db );
          void computeHT( const Patch* patch, HTVariables& vars, CCVariable<double>& T );
          void copySolution( const Patch* patch, CCVariable<double>& T, constCCVariable<double>& T_copy, constCCVariable<int>& cell_type );

        private:


          const double _sigma_constant;      ///< Stefan Boltzman constant [W/(m^2 K^4)]
          double _init_tol;                  ///< initial tolerance for the iterative solver
          double _tol;                       ///< solver tolerance
          int _max_it;                       ///< maximum iterations allowed

          struct WallInfo {
              double T_slag;
              double dy_erosion;
              double t_sb;
              double k;
              double k_deposit;
              double dy;
              double dy_dep_init; // initial deposit thickness
              double emissivity;
              double T_inner;
              double relax;     ///< A relaxation coefficient to help stability (eg, wall temperature changes too fast)...but not necessarily with accuracy
              double max_TW;     ///< maximum wall temperature
              double min_TW;     ///< minimum wall temperature
              std::vector<GeometryPieceP> geometry;
          };

          inline void newton_solve(WallInfo& wi, HTVariables& vars, double &TW_new, double &T_old, double &rad_q, double &net_q, double &R_tot, double &Emiss );

          std::vector<WallInfo> _regions;

          std::vector<IntVector> _d;

          inline constCCVariable<double>* get_flux( int i, HTVariables& vars ){

            constCCVariable<double>* q = NULL;
            switch (i) {
              case 0:
                q = &(vars.incident_hf_w);
                break;
              case 1:
                q = &(vars.incident_hf_e);
                break;
              case 2:
                q = &(vars.incident_hf_s);
                break;
              case 3:
                q = &(vars.incident_hf_n);
                break;
              case 4:
                q = &(vars.incident_hf_b);
                break;
              case 5:
                q = &(vars.incident_hf_t);
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
