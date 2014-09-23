#ifndef UT_DQMOMEqnFactory_h
#define UT_DQMOMEqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h> // should this be here?
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <map>
#include <vector>
#include <string>

//---------------------------------------------------------------------------
// Builder 

/**
  * @class DQMOMEqnBuilder
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date November 19, 2008
  *
  * @brief Abstract base class to support scalar equation additions.  Meant to
  * be used with the DQMOMEqnFactory. 
  *
  */
namespace Uintah {
class EqnBase; 
class DQMOMEqnBuilderBase
{
public:
  DQMOMEqnBuilderBase( ArchesLabel* fieldLabels, 
                       ExplicitTimeInt* timeIntegrator,
                       std::string eqnName ) :
                       d_fieldLabels(fieldLabels), 
                       d_eqnName(eqnName), 
                       d_timeIntegrator(timeIntegrator) {};

  virtual ~DQMOMEqnBuilderBase(){};

  virtual EqnBase* build() = 0;  

protected: 
  ArchesLabel* d_fieldLabels; 
  std::string d_eqnName;
  ExplicitTimeInt* d_timeIntegrator; 
}; // class DQMOMEqnBuilder

// End builder 
//---------------------------------------------------------------------------

/**
  * @class  DQMOMEqnFactory
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date   November 19, 2008
  * 
  * @brief  A Factory for building eqns. 
  * 
  */
class ArchesLabels; 
class ExplicitTimeInt; 
class DQMOMEqnFactory
{
public:

  enum NDF_DESCRIPTOR { TEMPERATURE, ENTHALPY, SIZE, MASS, COAL_MASS_FRAC, UVEL, VVEL, WVEL };

  typedef std::map< std::string, EqnBase* >     EqnMap; 

  /** @brief Return an instance of the factory.  */
  static DQMOMEqnFactory& self(); 

  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            DQMOMEqnBuilderBase* builder);

  /** @brief Register the DQMOM equations by parsing the input file. **/ 
  void registerDQMOMEqns(ProblemSpecP& db, ArchesLabel* field_labels, ExplicitTimeInt* time_integrator);

  /** @brief Register a weight eqn.    */
  void set_weight_eqn( const std::string name, EqnBase* eqn ); 

  /** @brief Register an abscissa eqn.    */
  void set_abscissa_eqn( const std::string name, EqnBase* eqn );

  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );

  /** @brief Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /** @brief Get access to the weights eqn map */
  EqnMap& retrieve_weights_eqns(){
    return weights_eqns_; };

  /** @brief Get access to the abscissas eqn map */
  EqnMap& retrieve_abscissas_eqns(){
    return abscissas_eqns_; };

  /** @brief Set number quadrature nodes */ 
  inline void set_quad_nodes( int qn ) {
    n_quad_ = qn; };

  /** @brief Get the number of quadrature nodes */ 
  inline int get_quad_nodes() {
    return n_quad_; };

  /** @brief Get the base string name given its NDF descriptor **/ 
  inline const std::string get_base_name( NDF_DESCRIPTOR desc ){ 

    std::map<NDF_DESCRIPTOR, std::string>::iterator iter = ndf_actors.find(desc); 
    if ( iter != ndf_actors.end() ){ 
      return iter->second; 
    }
    std::stringstream msg; 
    msg << "Error: Cannot find an actor matching this desciptor: " << desc << std::endl;
    throw InvalidValue( msg.str(),__FILE__,__LINE__);
    
  }

  /** @brief Assign the string name to a descriptor 
   *         Will not insert a duplicate copy **/ 
  void assign_descriptor( const std::string name, NDF_DESCRIPTOR desc ){ 

    std::map<NDF_DESCRIPTOR, std::string>::iterator iter = ndf_actors.find(desc); 
    if ( iter == ndf_actors.end() ){ 
      ndf_actors.insert(std::make_pair(desc, name));
    } else { 
      if ( iter->second != name ){ 
        std::stringstream msg; 
        msg << "Error: Trying to insert an DQMOM actor with a different name than one already registered." << std::endl;
        throw InvalidValue( msg.str(),__FILE__,__LINE__);
      }
    }
  }

  /** @brief Match string names of ndf_descriptors to the enum **/ 
  NDF_DESCRIPTOR get_descriptor( const std::string desc ){ 

    std::map<std::string, NDF_DESCRIPTOR>::iterator iter = string_to_ndf_desc.find(desc); 

    if ( iter != string_to_ndf_desc.end() ){
      return iter->second; 
    }
    std::stringstream msg; 
    msg << "Error: Cannot match string ndf descriptor with enum." << std::endl;
    throw InvalidValue( msg.str(),__FILE__,__LINE__);

  }

private:

  typedef std::map< std::string, DQMOMEqnBuilderBase* >  BuildMap; 

  BuildMap builders_; 
  EqnMap eqns_; 
  EqnMap weights_eqns_;
  EqnMap abscissas_eqns_;

  DQMOMEqnFactory(); 
  ~DQMOMEqnFactory(); 

  int n_quad_; 

  bool doing_dqmom_; 

  std::string which_dmqom_;

  std::map<NDF_DESCRIPTOR, std::string> ndf_actors; 
  std::map<std::string, NDF_DESCRIPTOR> string_to_ndf_desc; 

}; // class DQMOMEqnFactory 
} // end namespace Uintah

#endif
