#ifndef Uintah_Component_Arches_ParticleTools_h
#define Uintah_Component_Arches_ParticleTools_h

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>
#include <sstream>
#include <map>

namespace Uintah{ namespace ArchesCore{


  enum PARTICLE_METHOD {DQMOM_METHOD, CQMOM_METHOD, LAGRANGIAN_METHOD};
  enum PARTICLE_ROLE { P_SIZE, P_XVEL, P_YVEL, P_ZVEL, P_TEMPERATURE, P_ENTHALPY, P_TOTNUM_DENSITY,
                       P_MAXTEMPERATURE, P_RAWCOAL, P_CHAR, P_DENSITY, P_DTDT };

  /** @brief Parse for a role -> label match in the EulerianParticles section **/
  std::string parse_for_particle_role_to_label( ProblemSpecP& db, PARTICLE_ROLE role );

  /** @brief Map role string to enum **/
  std::string get_particle_role_from_enum( PARTICLE_ROLE role );

  /** @brief Obtain a vector<string> of all internal internal cordinates **/
  std::vector<std::string> getICNames( ProblemSpecP& db );

  /** @brief Append an environment to a string **/
  std::string append_env( std::string in_name, const int qn );

  /** @brief Append an qn environment to a string **/
  std::string append_qn_env( std::string in_name, const int qn );

  /** @brief Check for a particle method in the input file **/
  bool check_for_particle_method( ProblemSpecP& db, PARTICLE_METHOD method );

  /** @brief Returns the number of environments for a specific particle method **/
  int get_num_env( ProblemSpecP& db, PARTICLE_METHOD method );

  /** @brief Returns model type given the name **/
  std::string get_model_type( ProblemSpecP& db, std::string model_name, PARTICLE_METHOD method );

  /** @brief Get the scaling constant for a particular internal coordinate and qn **/
  double get_scaling_constant(ProblemSpecP& db, const std::string labelName, const int qn);

  /** @brief What does this do? **/
  bool get_model_value(ProblemSpecP& db, const std::string labelName, const int qn, double &value);

  /** @brief This function is needed for specifying the mass flow inlet of particles **/
  double get_inlet_particle_density(ProblemSpecP& db);

  /** @brief This function is useful when specifying mass flow inlet of particles **/
  double get_inlet_particle_size(ProblemSpecP& db, const int qn );

  /** @brief Return a list of models asssociated with a particular IC **/
  std::vector<std::string> getICModels( ProblemSpecP& db, const std::string ic_name );

  /** @brief Return a map with model and type of model **/
  std::map<std::string, std::string> getModelTypeMap( ProblemSpecP& db, const std::string ic_name );

  /** @brief  **/
  std::string getModelNameByType( ProblemSpecP& db, const std::string ic_name, const std::string model_type );
  
 

}} //Uintah::ArchesCore

#endif
