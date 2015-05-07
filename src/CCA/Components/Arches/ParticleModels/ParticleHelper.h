#ifndef Uintah_Component_Arches_ParticleHelper_h
#define Uintah_Component_Arches_ParticleHelper_h

#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{ 

  class ParticleHelper { 


    public: 

      enum PARTICLE_METHOD {DQMOM, CQMOM, LAGRANGIAN}; 

      ParticleHelper(){}
      ~ParticleHelper(){}

      /** @brief Parse for a role -> label match in the EulerianParticles section **/
      inline static std::string parse_for_role_to_label( ProblemSpecP& db, const std::string role ){ 

        const ProblemSpecP params_root = db->getRootNode();
        if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles") ){ 

          if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables") ){ 

            const ProblemSpecP db_pvar = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("EulerianParticles")->findBlock("ParticleVariables");

            for ( ProblemSpecP db_var = db_pvar->findBlock("variable"); db_var != 0; db_var = db_var->findNextBlock("variable")){ 

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
            int N; 
            arches_root->findBlock("CQMOM")->require("QuadratureNodes",N); 
            return N; 
          } else { 
            throw ProblemSetupException("Error: DQMOM particle method not found.", __FILE__, __LINE__); 
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

          for ( ProblemSpecP m_db = db_models->findBlock("model"); m_db != 0; m_db = m_db->findNextBlock("model") ) {

            std::string curr_model_name;
            std::string curr_model_type; 
            m_db->getAttribute("label",curr_model_name);
            m_db->getAttribute("type", curr_model_type); 

            if ( model_name == curr_model_name ){ 
              return curr_model_type; 
            }
          }
        } else { 
          throw InvalidValue("Error: Not yet implemented for this particle method.",__FILE__,__LINE__); 
        }
        return "NULL"; 
      }

    private: 

  
  }; 

}

#endif
