#ifndef Uintah_Component_Arches_ParticleTools_h
#define Uintah_Component_Arches_ParticleTools_h

#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{ 

  class ParticleTools {


    public: 

      enum PARTICLE_METHOD {DQMOM, CQMOM, LAGRANGIAN}; 

      ParticleTools(){}
      ~ParticleTools(){}

      /** @brief Parse for a role -> label match in the EulerianParticles section **/
      inline static std::string parse_for_role_to_label( ProblemSpecP& db, const std::string role ){ 

        const ProblemSpecP params_root = db->getRootNode();
        if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles") ){ 

          if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables") ){ 

            const ProblemSpecP db_pvar = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables");

            for( ProblemSpecP db_var = db_pvar->findBlock("variable"); db_var != nullptr; db_var = db_var->findNextBlock("variable")){ 

              std::string label;
              std::string role_found; 
              db_var->getAttribute("label", label);
              db_var->getAttribute("role", role_found); 

              if ( role_found == role ){ 
                return label; 
              }

            }

            throw ProblemSetupException("Error: This Eulerian particle role not found: "+role, __FILE__, __LINE__); 

          } else { 

            throw ProblemSetupException("Error: No <EulerianParticles><ParticleVariables> section found.",__FILE__,__LINE__);      

          }
        } else { 

          throw ProblemSetupException("Error: No <EulerianParticles> section found in input file.",__FILE__,__LINE__);      

        }

      }

      /** @brief Append an environment to a string **/ 
      inline static std::string append_env( std::string in_name, const int qn ){ 

        std::string new_name = in_name;
        std::stringstream str_qn; 
        str_qn << qn; 
        new_name += "_";
        new_name += str_qn.str(); 
        return new_name; 
        
      }

      /** @brief Append an qn environment to a string **/ 
      inline static std::string append_qn_env( std::string in_name, const int qn ){ 

        std::string new_name = in_name;
        std::stringstream str_qn; 
        str_qn << qn; 
        new_name += "_qn";
        new_name += str_qn.str(); 
        return new_name; 
        
      }

      /** @brief Check for a particle method in the input file **/ 
      inline static bool check_for_particle_method( ProblemSpecP& db, PARTICLE_METHOD method ){ 

        const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");
       
        bool check = false; 
        if ( method == DQMOM ){ 

          if ( arches_root->findBlock("DQMOM") ){ 
            check = true; 
          }

        } else if ( method == CQMOM ){ 

          if ( arches_root->findBlock("CQMOM") ){ 
            check = true; 
          }

        } else if ( method == LAGRANGIAN ){ 

          if ( arches_root->findBlock("LagrangianParticles") ){ 
            check = true; 
          }

        } else { 
          throw ProblemSetupException("Error: Particle method type not recognized.", __FILE__, __LINE__); 
        }

        return check; 

      }

      /** @brief Returns the number of environments for a specific particle method **/ 
      static int get_num_env( ProblemSpecP& db, PARTICLE_METHOD method ){ 

        const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");
       
        if ( method == DQMOM ){ 

          if ( arches_root->findBlock("DQMOM") ){ 
            int N; 
            arches_root->findBlock("DQMOM")->require("number_quad_nodes",N); 
            return N; 
          } else { 
            throw ProblemSetupException("Error: DQMOM particle method not found.", __FILE__, __LINE__); 
          }

        } else if ( method == CQMOM ){ 

          if ( arches_root->findBlock("CQMOM") ){ 
            int N = 1;
            std::vector<int> N_i;
            arches_root->findBlock("CQMOM")->require("QuadratureNodes",N_i);
            for (unsigned int i = 0; i < N_i.size(); i++ ) {
              N *= N_i[i];
            }
            return N; 
          } else { 
            throw ProblemSetupException("Error: CQMOM particle method not found.", __FILE__, __LINE__); 
          }

        } else if ( method == LAGRANGIAN ){ 

          if ( arches_root->findBlock("LagrangianParticles") ){ 
            return 1; 
          } else { 
            throw ProblemSetupException("Error: DQMOM particle method not found.", __FILE__, __LINE__); 
          }

        } else { 
          throw ProblemSetupException("Error: Particle method type not recognized.", __FILE__, __LINE__); 
        }
      }

      /** @brief Returns model type given the name **/ 
      static std::string get_model_type( ProblemSpecP& db, std::string model_name, PARTICLE_METHOD method ){ 

        const ProblemSpecP arches_root = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES");
       
        if ( method == DQMOM ){ 

          const ProblemSpecP db_models = arches_root->findBlock("DQMOM")->findBlock("Models"); 

          for ( ProblemSpecP m_db = db_models->findBlock("model"); m_db != nullptr; m_db = m_db->findNextBlock("model") ) {

            std::string curr_model_name;
            std::string curr_model_type; 
            m_db->getAttribute("label",curr_model_name);
            m_db->getAttribute("type", curr_model_type); 

            if ( model_name == curr_model_name ){ 
              return curr_model_type; 
            }
          }
        }
        else { 
          throw InvalidValue("Error: Not yet implemented for this particle method.",__FILE__,__LINE__); 
        }
        return "nullptr"; 
      }

      inline static double getScalingConstant(ProblemSpecP& db, const std::string labelName, const int qn){ 

        const ProblemSpecP params_root = db->getRootNode();
        if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM") ){ 

          if ( labelName == "w" ||  labelName == "weights" || labelName == "weight"){

            if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Weights") ){ 

              std::vector<double> scaling_const;
              params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Weights")->require("scaling_const",scaling_const);
              return scaling_const[qn];
            }
            else { 
              throw ProblemSetupException("Error: cannot find <weights> block in inupt file.",__FILE__,__LINE__);      
            }
          }
          else {
            for(  ProblemSpecP IcBlock =params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("DQMOM")->findBlock("Ic"); IcBlock != nullptr;
                  IcBlock = IcBlock->findNextBlock("Ic") ) {

              std::string tempLabelname;
              IcBlock->getAttribute("label",tempLabelname);
              if (tempLabelname == labelName){
                std::vector<double> scaling_const;
                IcBlock->require("scaling_const",scaling_const);
                return scaling_const[qn];
              }
            }
            throw ProblemSetupException("Error: couldn't find internal coordinate or weight with name: "+labelName , __FILE__, __LINE__); 
          } 
        }
        else { 
          throw ProblemSetupException("Error: DQMOM section not found in input file.",__FILE__,__LINE__);      
        }
      }


      inline static bool getModelValue(ProblemSpecP& db, const std::string labelName, const int qn, double &value){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels") ){
          const ProblemSpecP params_particleModels = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");
          for ( ProblemSpecP modelBlock =params_particleModels->findBlock("model"); modelBlock != nullptr;
              modelBlock = modelBlock->findNextBlock("model") ) {

            std::string tempLabelName;
            modelBlock->getAttribute("label",tempLabelName);
            if (tempLabelName == labelName){
              // -- expand this as different internal coordinate models are added--//
              std::string tempTypeName;
              modelBlock->getAttribute("type",tempTypeName);

              if( tempTypeName== "constant"){
                std::vector<double> internalCoordinateValue;
                modelBlock->require("constant", internalCoordinateValue);
                value= internalCoordinateValue[qn];
                return true;
              }
            }
          }
        }
        return false;
      }

      // This function is useful when specifying mass flow inlet of particles
      inline static double getInletParticleDensity(ProblemSpecP& db){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")){
        double density;
        params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("density",density);
        return density;
        }
        else {
          throw ProblemSetupException("Error: cannot find <ParticleProperties> in arches block.",__FILE__,__LINE__);      
        }

        return false;
      }

      inline static double getAshMassFraction(ProblemSpecP& db){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")){
          double ash_frac;
          double C;
          double H;
          double O;
          double N;
          double S;
          double CHAR;
          double ASH;
          double H2O;
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("ASH",ASH);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("C",C);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("H",H);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("O",O);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("N",N);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("S",S);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("CHAR",CHAR);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("ASH",ASH);
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ultimate_analysis")->require("H2O",H2O);
          ash_frac = ASH/(C+H+O+N+S+ASH+CHAR); // DRY ASH FRACTION!!!!!
          return ash_frac;
        } else{
          throw ProblemSetupException("Error: cannot find <ultimate_analysis> in arches block.",__FILE__,__LINE__);      
        }

        return false;
      }
      
      // This function is useful when specifying mass flow inlet of particles
      inline static double getInletParticleSize(ProblemSpecP& db, const int qn ){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("diameter_distribution")){
          std::vector<double> _sizes;
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("diameter_distribution", _sizes);
          double initial_size = _sizes[qn];
          return initial_size;
        } else{
          throw ProblemSetupException("Error: cannot find <diameter_distribution> in arches block.",__FILE__,__LINE__);      
        }

        return false;
      }
      
      // This function retrieves the ash hemispherical melting temperature for use in computing the melting probability in the deposition model.
      inline static double getAshHemisphericalTemperature(ProblemSpecP& db){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ash_hemispherical_temperature")){
          double T_hemisphere;
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("ash_hemispherical_temperature",T_hemisphere);
          return T_hemisphere;
        }
        else {
          throw ProblemSetupException("Error: cannot find <ash_hemispherical_temperature> in <ParticleProperties> in arches block.",__FILE__,__LINE__);      
        }
        return false;
      }
      
      // This function retrieves the average temperature between the softening and fluid temperatures for use in the porosity model for ash thermal conductivity.
      inline static double getAshPorosityTemperature(ProblemSpecP& db){ 

        const ProblemSpecP params_root = db->getRootNode();
        double T_ave;
        double T_soft;
        double T_fluid;
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ash_fluid_temperature")){
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("ash_fluid_temperature",T_fluid);
        }
        else {
          throw ProblemSetupException("Error: cannot find <ash_fluid_temperature> in <ParticleProperties> in arches block.",__FILE__,__LINE__);      
        }
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ash_softening_temperature")){
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("ash_softening_temperature",T_soft);
        }
        else {
          throw ProblemSetupException("Error: cannot find <ash_softening_temperature> in <ParticleProperties> in arches block.",__FILE__,__LINE__);      
        }
        T_ave = (T_soft + T_fluid) / 2.0;
        return T_ave;
      }
      
      // This function retrieves the fluid temperature for use as the slagging temeprature in the wallHT model.
      inline static double getAshFluidTemperature(ProblemSpecP& db){ 

        const ProblemSpecP params_root = db->getRootNode();
        if(params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->findBlock("ash_fluid_temperature")){
          double T_fluid;
          params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties")->require("ash_fluid_temperature",T_fluid);
          return T_fluid;
        }
        else {
          throw ProblemSetupException("Error: cannot find <ash_fluid_temperature> in <ParticleProperties> in arches block.",__FILE__,__LINE__);      
        }
        return false;
      }



    private: 
  
  }; 

}

#endif
