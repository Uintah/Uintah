#ifndef Uintah_Component_Arches_CoalHelper_h
#define Uintah_Component_Arches_CoalHelper_h

#include <Core/Exceptions/ProblemSetupException.h>

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
          if (particleType != "coal"){
            throw InvalidValue("ERROR in CoalHelper: I didn't find coal particles, I found particles of type: "+particleType,__FILE__,__LINE__);
          }

          db_coal_props->require("density",_coal_db.rhop_o); 
          db_coal_props->require("diameter_distribution", _coal_db.sizes); 

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
            double raw_coal_mf = coal_daf / coal_dry; 
            double char_mf = _coal_db.coal.CHAR / coal_dry; 
            double ash_mf = _coal_db.coal.ASH / coal_dry; 

            _coal_db.init_char.clear(); 
            _coal_db.init_rawcoal.clear(); 
            _coal_db.init_ash.clear(); 
            _coal_db.denom.clear(); 

            _coal_db.Nenv = _coal_db.sizes.size(); 

            for ( unsigned int i = 0; i < _coal_db.sizes.size(); i++ ){ 

              double mass_dry = (pi/6.0) * pow(_coal_db.sizes[i],3) * _coal_db.rhop_o;     // kg/particle
              _coal_db.init_ash.push_back(mass_dry  * ash_mf);                      // kg_ash/particle (initial)  
              _coal_db.init_char.push_back(mass_dry * char_mf);                     // kg_char/particle (initial)
              _coal_db.init_rawcoal.push_back(mass_dry * raw_coal_mf);              // kg_ash/particle (initial)
              _coal_db.denom.push_back( _coal_db.init_ash[i] + 
                  _coal_db.init_char[i] + 
                  _coal_db.init_rawcoal[i] );

            }

          } else { 
            throw ProblemSetupException("Error: No <ultimate_analysis> found in input file.", __FILE__, __LINE__); 
          }
        } else { 
          throw ProblemSetupException("Error: <Coal> is turned on but the <ParticleProperties> node is not found.", __FILE__, __LINE__); 
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
