#ifndef Uintah_Component_Arches_CoalHelper_h
#define Uintah_Component_Arches_CoalHelper_h

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <vector>

namespace Uintah{

  class CoalHelper {


    public:

      /** @brief Return the CoalHelper **/
      static CoalHelper& self(){
        static CoalHelper s;
        return s;
      }

      /** @brief An object to hold the coal db information **/
      struct CoalDBInfo{
        public:
        int Nenv;
        double value;
        double rhop_o;
        double pi;
        double raw_coal_mf;
        double char_mf;
        double ash_mf;
        double mw_avg;
        double h_c0;
        double h_ch0;
        double h_a0;
        double ksi;
        double Tar_fraction;
        double T_hemisphere;        ///< Ash hemispherical temperature
        double T_fluid;             ///< Ash fluid temperature
        double T_soft;              ///< Ash softening temperature
        double T_porosity;          ///< Ash porosity temperature
        double visc_pre_exponential_factor; ///< Ash viscosity pre-exponential factor [poise/K] -Urbain viscosity model
        double visc_activation_energy; ///< Ash viscosity pre-exponential factor [poise/K] -Urbain viscosity model

        std::vector<double> init_ash;
        std::vector<double> init_rawcoal;
        std::vector<double> init_char;
        std::vector<double> sizes;
        std::vector<double> denom;

        std::string rawcoal_base_name;
        std::string char_base_name;

        struct CoalAnalysis{
          double C;
          double H;
          double O;
          double N;
          double S;
          double CHAR;
          double ASH;
          double H2O;
        };

        CoalAnalysis coal;

      };

      /** @brief Parse coal information for use later **/
      void parse_for_coal_info( ProblemSpecP& db ){

        const ProblemSpecP db_root = db->getRootNode();

        double pi = acos(-1.0);
        if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){

          ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

          std::string particleType;
          db_coal_props->getAttribute("type",particleType);
          if ( particleType == "coal" ){

            db_coal_props->require("density",_coal_db.rhop_o);
            db_coal_props->require("diameter_distribution", _coal_db.sizes);
            db_coal_props->require("raw_coal_enthalpy", _coal_db.h_c0);
            db_coal_props->require("char_enthalpy", _coal_db.h_ch0);
            db_coal_props->require("ash_enthalpy", _coal_db.h_a0);
            db_coal_props->getWithDefault( "ksi",_coal_db.ksi,1.0); // Fraction of the heat released by char oxidation that goes to the particle
            db_coal_props->getWithDefault( "Tar_fraction",_coal_db.Tar_fraction,0.0); // Fraction of devol rate products that go to tar vs light off gases

            if ( db_coal_props->findBlock("ultimate_analysis")){

              //<!-- as received mass fractions C+H+O+N+S+char+ash+moisture=1 -->
              ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis");

              db_ua->require("C",_coal_db.coal.C);
              db_ua->require("H",_coal_db.coal.H);
              db_ua->require("O",_coal_db.coal.O);
              db_ua->require("N",_coal_db.coal.N);
              db_ua->require("S",_coal_db.coal.S);
              db_ua->require("H2O",_coal_db.coal.H2O);
              db_ua->require("ASH",_coal_db.coal.ASH);
              db_ua->require("CHAR",_coal_db.coal.CHAR);

              double coal_daf = _coal_db.coal.C + _coal_db.coal.H
                + _coal_db.coal.O + _coal_db.coal.N + _coal_db.coal.S; //dry ash free coal
              double coal_dry = _coal_db.coal.C + _coal_db.coal.H
                + _coal_db.coal.O + _coal_db.coal.N + _coal_db.coal.S
                + _coal_db.coal.ASH + _coal_db.coal.CHAR; //moisture free coal
              _coal_db.raw_coal_mf = coal_daf / coal_dry;
              _coal_db.char_mf = _coal_db.coal.CHAR / coal_dry;
              _coal_db.ash_mf = _coal_db.coal.ASH / coal_dry;

              _coal_db.init_char.clear();
              _coal_db.init_rawcoal.clear();
              _coal_db.init_ash.clear();
              _coal_db.denom.clear();

              _coal_db.Nenv = _coal_db.sizes.size();

              for ( unsigned int i = 0; i < _coal_db.sizes.size(); i++ ){

                double mass_dry = (pi/6.0) * pow(_coal_db.sizes[i],3) * _coal_db.rhop_o;     // kg/particle
                _coal_db.init_ash.push_back(mass_dry  * _coal_db.ash_mf);                    // kg_ash/particle (initial)
                _coal_db.init_char.push_back(mass_dry * _coal_db.char_mf);                   // kg_char/particle (initial)
                _coal_db.init_rawcoal.push_back(mass_dry * _coal_db.raw_coal_mf);            // kg_ash/particle (initial)
                _coal_db.denom.push_back( _coal_db.init_ash[i] +
                    _coal_db.init_char[i] +
                    _coal_db.init_rawcoal[i] );

              }
              _coal_db.pi = pi;

              double yElem [5];
              yElem[0]=_coal_db.coal.C/coal_daf; // C daf
              yElem[1]=_coal_db.coal.H/coal_daf; // H daf
              yElem[2]=_coal_db.coal.N/coal_daf; // N daf
              yElem[3]=_coal_db.coal.O/coal_daf; // O daf
              yElem[4]=_coal_db.coal.S/coal_daf; // S daf

              double MW [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S) - kg/kmol
              double mw_avg = 0.0; // Mean atomic weight of coal
              for(int i=0;i<5;i++){
                mw_avg += yElem[i]/MW[i];
              }
              _coal_db.mw_avg = 1.0/mw_avg;

            } else {
              throw ProblemSetupException("Error: No <ultimate_analysis> found in input file.", __FILE__, __LINE__);
            }

            //Ash temperatures:
            db_coal_props->getWithDefault("ash_hemispherical_temperature", _coal_db.T_hemisphere, -999);
            db_coal_props->getWithDefault("ash_fluid_temperature", _coal_db.T_fluid, -999);
            db_coal_props->getWithDefault("ash_softening_temperature", _coal_db.T_soft, -999);
            db_coal_props->getWithDefault("visc_pre_exponential_factor", _coal_db.visc_pre_exponential_factor, -999);
            db_coal_props->getWithDefault("visc_activation_energy", _coal_db.visc_activation_energy, -999);
            _coal_db.T_porosity = 0.5 * (_coal_db.T_soft + _coal_db.T_fluid);

          }
        }
      }

      CoalDBInfo& get_coal_db(){ return _coal_db; }

    private:

      CoalHelper(){}
      ~CoalHelper(){}

      CoalDBInfo _coal_db;

  };
}
#endif
