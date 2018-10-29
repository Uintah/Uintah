#ifndef Uintah_Components_Arches_WallModelDriver_h
#define Uintah_Components_Arches_WallModelDriver_h

#include <Core/Grid/LevelP.h>
#include <Core/Util/DebugStream.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

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

      WallModelDriver( MaterialManagerP& materialManager );
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
        double relax; // relaxation coefficient for updating surface temperature
        int em_model_type;
        int num_extra_src;
        int Nenv;
        double delta_t;
        double k_ash_uncertain;
        CCVariable<double> T;
        CCVariable<double> T_copy;
        CCVariable<double> T_real;
        CCVariable<double> deposit_thickness;
        CCVariable<double> deposit_thickness_sb_s;
        CCVariable<double> deposit_thickness_sb_l;
        CCVariable<double> emissivity;
        CCVariable<double> thermal_cond_en;
        CCVariable<double> thermal_cond_sb_s;
        CCVariable<double> thermal_cond_sb_l;
        CCVariable<double> deposit_velocity;
        constCCVariable<double> deposit_velocity_old;
        constCCVariable<double> ave_deposit_velocity;
        std::vector< constCCVariable<double> > extra_src;
        std::vector< constCCVariable<double> > particle_flow_rate;
        std::vector< constCCVariable<double> > particle_flow_rate_d;
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
        constCCVariable<double> deposit_thickness_sb_s_old;
        constCCVariable<double> deposit_thickness_sb_l_old;
        constCCVariable<double> emissivity_old;
        constCCVariable<double> thermal_cond_en_old;
        constCCVariable<double> thermal_cond_sb_s_old;
        constCCVariable<double> thermal_cond_sb_l_old;
        CCVariable<Stencil7> total_hf;
        constCCVariable<Vector > cc_vel;
        WallModelDriver::RAD_MODEL_TYPE model_type;

      };

    private:
      double _relax; // this is the global relaxation coefficient for any wallht type.
      std::string _dep_vel_name;
      std::vector<std::string> _extra_src_flux_names;
      int _num_extra_src;
      double _k_ash_uncertain;
      int _Nenv;
      bool do_coal_region;
      int _calc_freq;                    ///< Wall heat transfer model calculation frequency
      std::string _T_label_name;         ///< Temperature label name
      MaterialManagerP& _materialManager;
      int m_matl_index;                   ///< Material index

      const VarLabel* _timeStepLabel;
      const VarLabel* _simulationTimeLabel;
      const VarLabel* _delTLabel;

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
      std::vector< const VarLabel*> _extra_src_varlabels;
      std::vector< const VarLabel*> _particle_flow_rate_varlabels;
      std::vector< const VarLabel*> _particle_flow_rate_d_varlabels;
      const VarLabel* _deposit_velocity_label;
      const VarLabel* _deposit_thickness_label;
      const VarLabel* _deposit_thickness_sb_s_label;
      const VarLabel* _deposit_thickness_sb_l_label;
      const VarLabel* _emissivity_label;
      const VarLabel* _thermal_cond_en_label;
      const VarLabel* _thermal_cond_sb_s_label;
      const VarLabel* _thermal_cond_sb_l_label;

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
      inline static double get_k_ash_uncertain( const ProblemSpecP& input_db ){
        ProblemSpecP db = input_db;
        double k_val;
        db->getWithDefault( "k_ash_sigma", k_val,0.0);
        return k_val;
      }

      inline static std::vector<std::string> get_extra_src_flux_names( const ProblemSpecP& input_db ){
        ProblemSpecP db = input_db;
        std::vector<std::string> new_src_names;
        for ( ProblemSpecP db_src = db->findBlock( "extra_src" ); db_src != nullptr; db_src = db_src->findNextBlock( "extra_src" ) ){
          //get reaction rate params
          std::string new_src;
          new_src = db_src->getNodeValue();
          new_src_names.push_back(new_src);
        }
        return new_src_names;
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
        } else if ( model_type=="pokluda"){
          model_int = 3;
        } else if (model_type=="pokludaVisc"){
          model_int = 4;
        } else {
          throw InvalidValue("Error: emissivity_model_type must be either constant, dynamic, pokluda or pokludaVisc.", __FILE__, __LINE__);
        }
        return model_int;
      }

      inline static int get_thermal_cond_model_type( const ProblemSpecP& input_db ){
        ProblemSpecP db = input_db;
        std::string model_type;
        int model_int;
        db->getWithDefault( "thermal_cond_model_type", model_type,"constant");
        if (model_type=="constant"){
          model_int = 1;
        } else if ( model_type=="hadley"){
          model_int = 2;
        } else {
          throw InvalidValue("Error: thermal_cond_model_type must be either constant or hadley.", __FILE__, __LINE__);
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


          struct ThermalCondBase {
            virtual void model( double &k_eff, const double &C, const double &T,
                const std::string layer_type )=0;
            virtual ~ThermalCondBase(){}
          };

          ThermalCondBase* m_tc_model;

          struct constant_tc : ThermalCondBase {
            void model(double &k_eff, const double &C, const double &T, const std::string layer_type) {
              k_eff = C;
            }
            ~constant_tc(){}
          };

          struct hadley_tc : ThermalCondBase {
            hadley_tc(ProblemSpecP db_model){
              CoalHelper& coal_helper = CoalHelper::self();
              std::vector<double> default_comp = {39.36,25.49, 7.89,  10.12, 2.46, 0.0, 1.09, 4.10};
              db_model->getWithDefault( "enamel_ash_composition", en_ash_comp, default_comp);
              db_model->getWithDefault( "sb_ash_composition", sb_ash_comp, default_comp);
              db_model->getWithDefault( "enamel_deposit_porosity", en_porosity, 0.6);
              db_model->getWithDefault( "sb_deposit_porosity", sb_porosity, 0.6);
	            //T_mid = coal_helper.get_coal_db().T_porosity;
	            T_fluid = coal_helper.get_coal_db().T_fluid;
              if (en_ash_comp.size() != 8 || sb_ash_comp.size() != 8){
                throw InvalidValue("Error ash_compositions (enamel_ash_composition and sb_ash_composition) must have 8 entries: sio2, al2o3, cao, fe2o3, na2o, bao, tio2, mgo. ", __FILE__, __LINE__);
              }

              double poly_datas[6][5] = {{2.511097e1, -7.293704e-2, 9.210643e-5, -5.005416e-8, 9.748495e-12},//%sio2
                {2.656406e1, -5.013363e-2, 4.798874e-5, -2.274117e-8, 4.353314e-12},//%al2o3
                {2.738097e1, -9.707191e-2, 1.303015e-4, -6.99833e-8 , 1.467744e-11},//fe2o3
                {1.79954e1 , -2.807481e-2, 2.769034e-5, -1.244762e-8, 2.104583e-12},//cao
                {8.320303e1, -1.780395e-1, 1.598474e-4, -6.622821e-8, 1.072425e-11},//mgo
                {3.03673e1 , -6.10737e-2 , 6.65477e-5 , -3.40088e-8 , 6.17558e-12}};//na2o
              for (int i=0; i<6; ++i) {
                for (int j=0; j<5; ++j) {
                  poly_data[i][j] = poly_datas[i][j];
                }
              }
              en_ash_comp_tc = {en_ash_comp[0],en_ash_comp[1],en_ash_comp[3],en_ash_comp[2],en_ash_comp[7],en_ash_comp[4]};
              sb_ash_comp_tc = {sb_ash_comp[0],sb_ash_comp[1],sb_ash_comp[3],sb_ash_comp[2],sb_ash_comp[7],sb_ash_comp[4]};
              double en_ash_comp_tc_sum = 0.0;
              double sb_ash_comp_tc_sum = 0.0;
              std::for_each(en_ash_comp_tc.begin(), en_ash_comp_tc.end(), [&] (double n) { en_ash_comp_tc_sum += n;});
              std::for_each(sb_ash_comp_tc.begin(), sb_ash_comp_tc.end(), [&] (double n) { sb_ash_comp_tc_sum += n;});
              for (int i=0; i<6; ++i) {// renormalize to 1.
                en_ash_comp_tc[i] = en_ash_comp_tc[i]/en_ash_comp_tc_sum;
                sb_ash_comp_tc[i] = sb_ash_comp_tc[i]/sb_ash_comp_tc_sum;
              }
              double pb = 0.5;
              f0 = 0.8+0.1*pb;
            }
            std::vector<double> en_ash_comp_tc;
            std::vector<double> sb_ash_comp_tc;
            std::vector<double> ash_comp_tc;
            std::vector<double> en_ash_comp;
            std::vector<double> sb_ash_comp;
            double f0;
            double en_porosity;
            double sb_porosity;
            double T_fluid;
            double poly_data[6][5];
            void model(double &k_eff, const double &C, const double &T, const std::string layer_type) {
              ash_comp_tc = (layer_type == "enamel") ? en_ash_comp_tc : sb_ash_comp_tc;
              double ks, k, kg, a, kappa;
              std::vector<double> ki;

              // first compute solid tc as a function of temperature
              for (int i=0; i<6; ++i) {
                k = poly_data[i][0] + poly_data[i][1]*T + poly_data[i][2]*T*T + poly_data[i][3]*T*T*T + poly_data[i][4]*T*T*T*T;
                ki.push_back(k);
              }
              ks=0.0;
              for (int i=0; i<6; ++i) {
                ks=ks+ki[i]*ash_comp_tc[i];
              }
              // second compute the gas tc as a function of temperature
              kg = 2.286e-11*T*T*T - 7.022e-8*T*T + 1.209e-4*T - 5.321e-3;

              double phi; // this is a zeroth order porosity model for the sootblow layer
              phi = (layer_type == "enamel") ? en_porosity :  // enamel layer porosity never changes.
                (T > T_fluid) ? 0.0 : // if this is the sb layer get porosity based on T.
                sb_porosity;
              // third compute effective k for layer using hadley model
              a = (phi>=0.3) ? 1.5266*std::pow(1-phi,8.7381) : 0.7079*std::pow(1-phi,6.3051);
              kappa = ks/kg;
              k_eff = kg*((1.0-a)*(phi*f0 + (1.0-phi*f0)*kappa)/(1.0-phi*(1.0-f0) + phi*(1.0-f0)*kappa) + a*(2.0*(1.0-phi)*kappa*kappa + (1.0+2.0*phi)*kappa)/((2.0+phi)*kappa + 1.0 - phi));
            }
            ~hadley_tc(){}
          };


          struct EmissivityBase {
            virtual void model(double &e, const double &C, double &T, double &Dp, double &tau, std::vector<double> Dp_vec, std::vector<double> vdot)=0;
            virtual ~EmissivityBase(){}};

          EmissivityBase* m_em_model;

          struct constant_e : EmissivityBase {
            void model(double &e, const double &C, double &T, double &Dp, double &tau, std::vector<double> Dp_vec, std::vector<double> vdot) {
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
              } else if (ash_type == "german_lignite"){
                a_sv = -1.28417626e4;
                b_sv = -1.03858154;
                c_sv = 4.73690552;
                a_agg = 3.32945977e-3;
                b_agg = 8.57489019;
                c_agg = 3.31450986e-1;
                dp_eff_max = 0.67474660;
                dp_eff_min = 0.33333333;
                dpmax = 3000*1e-6;
                coeff_num = {0.13791672, -0.14171277, 0.45335828, 0.34291611, 0.15374553, -0.41985056};
                coeff_den = {1.00000000, 0.01302879, 0.25877226, -0.02239544, 0.01016679, -0.63044853};
                xscale = {750, 0.001496250000000};
                xcenter = {1050, 0.00150375};
                yscale = 8.97627710e-2;
                ycenter = 0.927990005;
                fresnel={0.966128001, -1.14864873e-06, -0.164711924, -0.00571174902};
              } else if (ash_type == "illinois_6"){
                a_sv = -1.81261708e+04;
                b_sv = -1.02046558e+00;
                c_sv = 6.31862841e+00;
                a_agg = 3.32945976e-03;
                b_agg = 8.57489020e+00;
                c_agg = 3.31450986e-01;
                dp_eff_max=0.67474660;
                dp_eff_min=0.33333333;
                dpmax = 0.00300000;
                coeff_num = {0.16547248, -0.07599717, 1.33842831, 0.43532097, 0.56769141, 0.15716789};
                coeff_den = {1.00000000, 0.37559090, 0.79324597, -0.01774033, 0.31671219, 0.08076074};
                xscale = {7.50000000e+02, 1.46250000e-04};
                xcenter = {1.05000000e+03, 1.53750000e-04};
                yscale = 1.26426850e-01;
                ycenter = 7.98292921e-01;
                fresnel={9.62350244e-01, -1.32310169e-06, -1.56147490e-01, -5.09003099e-03};
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
            void model(double &e, const double &C, double &T, double &Dp, double &tau, std::vector<double> Dp_vec, std::vector<double> vdot) {

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

          struct pokluda_e : EmissivityBase {
            pokluda_e(ProblemSpecP db_model){
              ProblemSpecP db_root = db_model->getRootNode();
              int Nenv = ArchesCore::get_num_env( db_root, ArchesCore::DQMOM_METHOD );
              CoalHelper& coal_helper = CoalHelper::self();
              min_p_diam = 1e16;
              double min_p = min_p_diam;
              double rho_ash_bulk;
              double p_void0;
              double v_hiT;
              for(int i = 0; i != Nenv; i++) {
                  ProblemSpecP db_part_properties = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
                  if (db_part_properties->findBlock("FOWYDevol")) {
                    ProblemSpecP db_BT = db_part_properties->findBlock("FOWYDevol");
                    db_BT->require("v_hiT", v_hiT); //
                  } else {
                    throw ProblemSetupException("Error: CharOxidationSmith2016 requires FOWY v_hiT.", __FILE__, __LINE__);
                  }
                  db_part_properties->getWithDefault( "rho_ash_bulk",rho_ash_bulk,2300.0);
                  db_part_properties->getWithDefault( "void_fraction",p_void0,0.3);
                  double init_particle_density = ArchesCore::get_inlet_particle_density( db_root );
                  double ash_mass_frac = coal_helper.get_coal_db().ash_mf;
                  double initial_diameter = ArchesCore::get_inlet_particle_size( db_root, i );
                  double p_volume = M_PI/6.*initial_diameter*initial_diameter*initial_diameter; // particle volme [m^3]
                  double mass_ash = p_volume*init_particle_density*ash_mass_frac;
                  double initial_rc = (M_PI/6.0)*initial_diameter*initial_diameter*initial_diameter*init_particle_density*(1.-ash_mass_frac);
                  double rho_org_bulk = initial_rc / (p_volume*(1-p_void0) - mass_ash/rho_ash_bulk) ; // bulk density of char [kg/m^3]
                  double p_voidmin = 1. - (1/p_volume)*(initial_rc*(1.-v_hiT)/rho_org_bulk + mass_ash/rho_ash_bulk); // bulk density of char [kg/m^3]
                  min_p = std::pow( mass_ash * 6 / rho_ash_bulk / (1- p_voidmin) / M_PI ,1./3.);
                  min_p_diam = (min_p < min_p_diam) ? min_p : min_p_diam;
              }
              if (min_p_diam == 1e16) {
                throw InvalidValue("Error, WallHT pokluda emissivity requires Particle Properties void fraction and rho ash or inlet particle size.", __FILE__, __LINE__);
              }
              std::string ash_type;
              db_model->getWithDefault( "coal_name", ash_type, "generic_coal");
              db_model->getWithDefault( "coordination_number", CN, 2);
	            T_fluid = coal_helper.get_coal_db().T_fluid;
              if (ash_type == "indonesian"){
                a_sv = -1.47871222e+04;
                b_sv = -1.02820241e+00;
                c_sv = 5.42161355e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779744e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {0.05524317, -0.34726897, 1.38639835, 0.38609096, 0.75315579, 0.12592464};
                coeff_den = {1.00000000, 0.40634839, 0.72008722, 0.04182158, 0.30286951, 0.15399492};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.39925847e-01;
                ycenter = 7.51584220e-01;
                pokluda={1.25909498e+00, 1.00031376e-05, -2.58931875e-01, -9.49645841e-01};
                fresnel={9.67088442e-01, -1.10250309e-06, -1.60617521e-01, -5.52026136e-03};
              } else if (ash_type == "german_lignite"){
                a_sv = -1.25814748e+04;
                b_sv = -1.03480466e+00;
                c_sv = 4.79358672e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779744e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {-0.19748923, -0.81370772, 0.96837919, 0.93623968, 0.32258380, -0.42526438};
                coeff_den = {1.00000000, 0.01012817, 0.16863730, -0.02276330, 0.00977648, -0.24310889};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.43440734e-01;
                ycenter = 7.24374513e-01;
                pokluda={1.25909498e+00, 1.00031376e-05, -2.58931875e-01, -9.49645841e-01};
                fresnel={9.66143598e-01, -1.16070664e-06, -1.64373764e-01, -5.70513057e-03};
              } else if (ash_type == "sufco"){
                a_sv = -1.33958768e+04;
                b_sv = -1.02934672e+00;
                c_sv = 5.07798702e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779744e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {0.02216238, -0.87550219, 1.27350751, 0.03491962, 0.10608581, 0.48838827};
                coeff_den = {1.00000000, 0.54029083, 0.92142994, 0.09869234, 0.36935055, 0.23640155};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.53758659e-01;
                ycenter = 6.83145516e-01;
                pokluda={1.25909498e+00, 1.00031376e-05, -2.58931875e-01, -9.49645841e-01};
                fresnel={9.63652299e-01, -1.77564557e-06, -1.67503707e-01, -5.48407743e-03};
              } else if (ash_type == "black_thunder"){
                a_sv = -1.18467478e+04;
                b_sv = -1.03834999e+00;
                c_sv = 4.63029656e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779744e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {-0.25727860, -0.91104953, 0.88695805, 0.92690861, 0.23159290, -0.33016112};
                coeff_den = {1.00000000, 0.00872138, 0.15822618, -0.03010546, 0.00841043, -0.22825951};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.53882421e-01;
                ycenter = 6.80899256e-01;
                pokluda={1.25909498e+00, 1.00031376e-05, -2.58931875e-01, -9.49645841e-01};
                fresnel={9.64255948e-01, -2.39444175e-06, -1.65069322e-01, -6.48251264e-03};
              } else if (ash_type == "illinois_no6"){
                a_sv = -1.79643791e+04;
                b_sv = -1.01881008e+00;
                c_sv = 6.35831792e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779745e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {0.24379045, -0.00534168, 1.48542122, 0.32731987, 0.75775571, 0.22750005};
                coeff_den = {1.00000000, 0.43600929, 0.99143155, 0.01901423, 0.37916811, 0.31903692};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.30498623e-01;
                ycenter = 7.99064834e-01;
                pokluda={1.25909498e+00, 1.00031388e-05, -2.58931875e-01, -9.49645843e-01};
                fresnel={9.62541957e-01, -1.46992243e-06, -1.53889119e-01, -5.03828096e-03};
              } else if (ash_type == "polish_coal"){
                a_sv = -1.48137127e+04;
                b_sv = -1.02726717e+00;
                c_sv = 5.43260270e+00;
                a_xr = 2.59027058e-01;
                b_xr = 2.07779745e+00;
                c_xr = 1.00036000e+00;
                coeff_num = {0.25498761, -0.05209599, 1.56617806, 0.28450416, 0.73540555, 0.36014319};
                coeff_den = {1.00000000, 0.42502868, 1.02554125, 0.01881639, 0.34867095, 0.32685695};
                xscale = {6.50000000e+02, 1.47500000e-04};
                xcenter = {9.50000000e+02, 1.52500000e-04};
                yscale = 1.29486006e-01;
                ycenter = 7.92863708e-01;
                pokluda={1.25909498e+00, 1.00031378e-05, -2.58931875e-01, -9.49645841e-01};
                fresnel={9.65180162e-01, -1.37020417e-06, -1.58188334e-01, -5.62284478e-03};
              } else {
                throw InvalidValue("Error, coal_name wasn't recognized in pokluda ash emissivity data-base. ", __FILE__, __LINE__);
              }
            }
            int CN; // coordination number
            double T_fluid;
            double a_sv;
            double b_sv;
            double c_sv;
            double a_xr;
            double b_xr;
            double c_xr;
            double min_p_diam;
            std::vector<double> coeff_num;
            std::vector<double> coeff_den;
            std::vector<double> xscale;
            std::vector<double> xcenter;
            double yscale;
            double ycenter;
            std::vector<double> fresnel;
            std::vector<double> pokluda;
            void model(double &e, const double &C, double &T, double &Dp, double &tau, std::vector<double> Dp_vec, std::vector<double> vdot) {

              // surface tension and viscosity model:
              // power law fit: log10(st/visc) = a*T^b+c
              double log10SurfT_div_Visc = a_sv*std::pow(T,b_sv)+c_sv; // [=] log10(m-s)
              double SurfT_div_Visc = std::pow(10,log10SurfT_div_Visc); // [=] m-s

              // non-dimensional time-scale
              double x_time =  SurfT_div_Visc*tau/(Dp/2.0);// [=] -

              // pokluda model for sintering
              // 2nd order exponential fit: d_eff = a*exp(b*X)+c*exp(d*X);
              double d_eff = (x_time>=100.0) ? (pokluda[0]*std::exp(pokluda[1]*100.0) + pokluda[2]*std::exp(pokluda[3]*100.0)) :
                (pokluda[0]*std::exp(pokluda[1]*x_time) + pokluda[2]*std::exp(pokluda[3]*x_time));
              double x_r = std::pow(std::max(d_eff - c_xr,0.0)/a_xr, 1.0/b_xr);
              d_eff = d_eff*Dp + (std::pow(CN,1./3.) - std::pow(2,1./3.))*x_r*Dp;// linear scale d_eff with respect to x_r
	      d_eff = std::max(5e-6,std::min(300e-6,d_eff)); // limit size to the bounds of the regression
              // as a function of the coordination number.
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
              e = (T>=T_fluid) ? ef :  // slagging is set to fresnel emissivity.
                (Dp<=min_p_diam) ? C : // if not slagging than use wall emissivity if flux is small.
                std::min(ef,e);  // if flux is positive you predicted emissivity.
	      e = std::max(0.0,std::min(1.0,e)); // emissivity safety
            }
            ~pokluda_e(){}
          };

          struct pokluda_v : EmissivityBase {
            pokluda_v(ProblemSpecP db_model){
              ProblemSpecP db_root = db_model->getRootNode();
              Nenv = ArchesCore::get_num_env( db_root, ArchesCore::DQMOM_METHOD );
              CoalHelper& coal_helper = CoalHelper::self();
              double rho_ash_bulk;
              double p_void0;
              double v_hiT;
              for(int i = 0; i != Nenv; i++) {
                ProblemSpecP db_part_properties = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
                if (db_part_properties->findBlock("FOWYDevol")) {
                  ProblemSpecP db_BT = db_part_properties->findBlock("FOWYDevol");
                  db_BT->require("v_hiT", v_hiT); //
                } else {
                  throw ProblemSetupException("Error: CharOxidationSmith2016 requires FOWY v_hiT.", __FILE__, __LINE__);
                }
                db_part_properties->getWithDefault( "rho_ash_bulk",rho_ash_bulk,2300.0);
                db_part_properties->getWithDefault( "void_fraction",p_void0,0.3);
              }
              std::string ash_type;
              db_model->getWithDefault( "coal_name", ash_type, "generic_coal");
	            T_fluid = coal_helper.get_coal_db().T_fluid;
	            double A = coal_helper.get_coal_db().visc_pre_exponential_factor;
              lnA = std::log(A);
	            B = coal_helper.get_coal_db().visc_activation_energy;
              if (A==-999 || B==-999){
                throw ProblemSetupException("Error: PokludaVisc requires specification of ash viscosity parameters.", __FILE__, __LINE__);
              }
              if (ash_type == "indonesian"){
		emiss_coeff = {0.42210472, -0.29718513, 0.52847460, -0.66785780, 0.36898482, -0.13952110, -0.00351870, 0.07376239, 0.20998621, -0.26403539, 0.07380761, 0.08677416, -0.05871292, 0.08181048, -0.07999015, 0.02236591, -0.03759233, 0.04154652, -0.03584659, 0.01244171, 0.00649477, -0.00330030, 0.00378924, 0.01575434, -0.00453514, 0.56640113, -0.00770423, -0.07205439};                
                xmean = {9.50000000e+02, 2.90527950e-04};
                xstd = {4.15013986e+02, 2.96789673e-04};
                ymean = 6.90652085e-01;
                ystd = 2.29605961e-01;
                surfT = {5.80176135e-01,-7.82543187e-05};
                pokluda={-4.77523025e-02, 1.26741073e+00};
                fresnel={9.67076543e-01, -1.09397153e-06, -1.60576835e-01, -5.52272718e-03};
              } else if (ash_type == "german_lignite"){
                emiss_coeff = {0.44599557, -0.30478529, 0.51486973, -0.61507071, 0.32793017, -0.19859028, 0.02383266, 0.07590513, 0.17344142, -0.23015802, 0.06493135, 0.11102434, -0.05095981, 0.06994493, -0.07454708, 0.02110421, -0.03606207, 0.03776718, -0.03850843, 0.01366592, 0.00309512, -0.00582621, 0.00553658, 0.01637788, -0.00343548, 0.52336435, -0.00782776, -0.06257533};                
                xmean = {9.50000000e+02, 2.90527950e-04};
                xstd = {4.15013986e+02, 2.96789673e-04};
                ymean = 6.69825177e-01;
                ystd = 2.32511918e-01;
                surfT = {5.71301560e-01,-8.13809376e-05};                
                pokluda={-4.77182710e-02, 1.26365363e+00};
                fresnel={9.66134618e-01, -1.15469725e-06, -1.64480775e-01, -5.70890613e-03};                
              } else if (ash_type == "sufco"){
                emiss_coeff = {0.47442630, -0.29968346, 0.47644786, -0.55391767, 0.29175602, -0.28389067, 0.05335562, 0.07140902, 0.13505376, -0.18947381, 0.05381933, 0.14227344, -0.04835121, 0.06211569, -0.07001087, 0.01996109, -0.03701543, 0.03863498, -0.04071644, 0.01457128, -0.00181829, -0.00665984, 0.00636716, 0.01892057, -0.00372373, 0.46096096, -0.00824010, -0.05533698};
                xmean = {9.50000000e+02, 2.90527950e-04};
                xstd = {4.15013986e+02, 2.96789673e-04};
                ymean = 6.42478139e-01;
                ystd = 2.31539137e-01;
                surfT = {4.99663849e-01,-4.78930294e-05};
                pokluda={-4.66118558e-02, 1.26945014e+00};
                fresnel={9.63806575e-01, -1.84039293e-06, -1.67140818e-01, -5.49005745e-03};
              } else if (ash_type == "black_thunder"){//River Basin
                emiss_coeff = {0.46529034, -0.29981580, 0.48505851, -0.56039858, 0.29329730, -0.28749424, 0.04915288, 0.08635075, 0.12343530, -0.18570493, 0.05327620, 0.14513120, -0.03758705, 0.05707612, -0.07099795, 0.02039353, -0.03542345, 0.03530547, -0.04478814, 0.01630458, -0.00151683, -0.01012663, 0.00822094, 0.01733052, -0.00222077, 0.46108133, -0.00777840, -0.05537377};
                xmean = {9.50000000e+02, 2.90527950e-04};
                xstd = {4.15013986e+02, 2.96789673e-04};
                ymean = 6.42437007e-01;
                ystd = 2.34747731e-01;
                surfT = {6.65251325e-01,-1.01475542e-04};
                pokluda={-4.76777178e-02, 1.26619132e+00};
                fresnel={9.64367875e-01, -2.43995449e-06, -1.65558767e-01, -6.51558072e-03};
              } else if (ash_type == "illinois_no6"){
                emiss_coeff = {0.28116197, -0.04796234, 0.44878875, -1.02871474, 0.70791090, -0.02938925, -0.11651905, 0.15951265, 0.27031796, -0.36712175, 0.10457104, 0.02109516, -0.03796620, 0.10692939, -0.11639365, 0.03423059, -0.02731009, 0.04638226, -0.04060291, 0.01387486, 0.01275071, -0.00413343, 0.00258171, 0.01089748, -0.00413871, 0.67618414, -0.00648648, -0.15473454};                
                xmean = {9.50000000e+02, 2.90527950e-04};
                xstd = {4.15013986e+02, 2.96789673e-04};
                ymean = 7.26042062e-01;
                ystd = 2.16751507e-01;
                surfT = {4.03907696e-01,-1.27139046e-05}; 
                pokluda={-5.14762513e-02, 1.26612415e+00};
                fresnel={9.58939398e-01, -1.94306615e-06, -1.52516593e-01, -5.40993057e-03};                
              } else if (ash_type == "polish_coal"){
                emiss_coeff = {0.48614013, -0.52506995, 0.67895992, -0.46501249, 0.13443785, -0.05815506, -0.08908543, 0.15646069, 0.23721181, -0.34574423, 0.10183021, 0.06647438, -0.07050489, 0.08788616, -0.07187695, 0.01902046, -0.04944071, 0.04988776, -0.02755827, 0.00713287, 0.01706380, -0.00479225, 0.00217505, 0.01252978, -0.00357129, 0.58671155, -0.00761589, -0.01163714};
                xmean = {9.50000000e+02, 1.66304348e-04};
                xstd = {4.15013986e+02, 1.63118380e-04};
                ymean = 7.03899649e-01;
                ystd = 2.04027127e-01;
                surfT = {4.20659229e-01,-2.36884087e-05}; 
                pokluda={-5.10314026e-02, 1.26264329e+00};
                fresnel={9.58939398e-01, -1.94306615e-06, -1.52516593e-01, -5.40993057e-03};                
              } else {
                throw InvalidValue("Error, coal_name wasn't recognized in pokluda ash emissivity data-base. ", __FILE__, __LINE__);
              }
            }
            int Nenv;
            double T_fluid;
            std::vector<double> coeff_num;
            std::vector<double> coeff_den;
            std::vector<double> xstd;
            std::vector<double> xmean;
            std::vector<double> pokluda; 
            std::vector<double> emiss_coeff;
            std::vector<double> fresnel;
            std::vector<double> surfT;
            double ystd;
            double log10tau;
            double visc;
            double lnA;
            double B;
            double surface_tension;
            double ymean;
            double vdot_sum;
            double d_eff;
            double x1;
            double x2;
            void model(double &e, const double &C, double &T, double &Dp, double &tau, std::vector<double> Dp_vec, std::vector<double> vdot) {
              e = 0;
              vdot_sum = 0; 
              // for each size compute the effective size and the effective emissivity
              double ef=fresnel[0]*std::exp(fresnel[1]*T) + fresnel[2]*std::exp(fresnel[3]*T);
              for(int env = 0; env != Nenv; env++) {
		            //// MILO
                //T = 1535.09;
                //tau = 1485.441;
                //Dp_vec[0] = 1.46777e-05;
                //Dp_vec[1] = 7.74869e-05;
                //Dp_vec[2] = 0.000130429;
                //Dp_vec[3] = 0.000235755;
                //Dp_vec[4] = 0.000340963;
                //vdot[0] = 2.11031e-10;
                //vdot[1] = 1.89357e-10;
                //vdot[2] = 1.11495e-10;
                //vdot[3] = 3.0919e-11;
                //vdot[4] = 2.36355e-11;
                //First obtain the effective particle size for the current environment.
                visc = 0.1*std::exp(std::log(T) + 1000.0*B/T + lnA); // kg / m / s
                surface_tension = surfT[0] + surfT[1]*T; // kg / s^2 
                log10tau = std::log10(tau*surface_tension/visc/Dp_vec[env]); // non-dimensionalized time
                d_eff =  Dp_vec[env] + 0.5*(Dp_vec[env]*std::cbrt(6.)-Dp_vec[env]) + 0.5*(Dp_vec[env]*std::cbrt(6.)-Dp_vec[env])*std::erf((log10tau-pokluda[0])/(std::sqrt(2.)*pokluda[1]));
		            d_eff = std::max(5e-6,std::min(400e-6,d_eff)); // limit size to the bounds of the regression
                //std::cout << "surface_tension " << surface_tension << std::endl;
                //std::cout << "visc " << visc << std::endl;
                //std::cout << "T: " << T << " tau: " << tau << " Dp_vec[env]: " << Dp_vec[env] << " d_eff: " << d_eff << std::endl; 
                //std::cout << "erf: " << (log10tau-pokluda[0])/(std::sqrt(2.)*pokluda[1])  << " " << std::erf((log10tau-pokluda[0])/(std::sqrt(2.)*pokluda[1])) << std::endl; 
                x1 = (T-xmean[0])/xstd[0];
                x2 = (d_eff-xmean[1])/xstd[1];
                double xv_e[28]={x2, x2*x2, x2*x2*x2, x2*x2*x2*x2, x2*x2*x2*x2*x2, x1, x1*x2, x1*x2*x2, x1*x2*x2*x2, x1*x2*x2*x2*x2, x1*x2*x2*x2*x2*x2, x1*x1, x1*x1*x2, x1*x1*x2*x2, x1*x1*x2*x2*x2, x1*x1*x2*x2*x2*x2, x1*x1*x1, x1*x1*x1*x2, x1*x1*x1*x2*x2, x1*x1*x1*x2*x2*x2, x1*x1*x1*x1, x1*x1*x1*x1*x2, x1*x1*x1*x1*x2*x2, x1*x1*x1*x1*x1, x1*x1*x1*x1*x1*x2, 1., x1*x1*x1*x1*x1*x1, x2*x2*x2*x2*x2*x2};
                double e_sc_c=0;
                for ( int I=0; I < 28; I++ ) {
                  e_sc_c+=emiss_coeff[I]*xv_e[I];
                }
                double e_i = e_sc_c*ystd + ymean; 
                e_i = (tau>=8640000.) ? C : // if no deposition use wall emissivity. This corresponds to a replacement time-scale of 100 days.
                    (T>=T_fluid || d_eff >= 400e-6) ? ef :  // if slagging use fresnel emissivity.
	                  std::min(ef,e_i);  // predicted emissivity.
                e += (vdot[env]+1e-100)*e_i; // sum in this environments emissivity weighted by flow rate
                //std::cout << "env " << env << std::endl;
                //std::cout << "e_i: " << e_i << std::endl; 
                vdot_sum+=vdot[env]+1e-100; // sum in this environments volumetric flow rate
              }
              e = e/vdot_sum;
              //std::cout << "T_fluid: " << T_fluid << std::endl;
              //std::cout << "d_eff: " << d_eff << std::endl;
	            e = std::max(0.0,std::min(1.0,e)); // emissivity safety
              //std::cout << "e after clipping: " << e << std::endl; 
            }
            ~pokluda_v(){}
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
            double Nenv;
            double dy_erosion;
            double t_sb;
            double k; // wall
            double dy; // wall
            double k_dep_en; // enamel deposit t.c.
            double k_dep_sb; // soot blown deposit t.c.
            double dy_dep_en; // enamel deposit thickness
            double emissivity;
            double T_inner;
            double max_TW;     ///< maximum wall temperature
            double min_TW;     ///< minimum wall temperature
            double deposit_density;
            std::vector<double> x_ash;
            std::vector<GeometryPieceP> geometry;
          };

          inline void newton_solve(double &TW_new, double &T_shell, double &T_old, double &R_tot, double &rad_q, double &Emiss, double &extra_src );
          inline void urbain_viscosity(double &visc, double &T, std::vector<double> &x_ash);

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
