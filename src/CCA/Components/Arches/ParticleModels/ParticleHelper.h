#ifndef Uintah_Component_Arches_ParticleHelper_h
#define Uintah_Component_Arches_ParticleHelper_h

#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah{ 

  class ParticleHelper { 


    public: 

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

    private: 

  
  }; 

}

#endif
