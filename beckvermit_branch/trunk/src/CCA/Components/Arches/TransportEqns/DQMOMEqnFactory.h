#ifndef UT_DQMOMEqnFactory_h
#define UT_DQMOMEqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h> // should this be here?
#include <Core/Grid/Variables/VarLabel.h>
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
                       string eqnName ) : 
                       d_fieldLabels(fieldLabels), 
                       d_eqnName(eqnName), 
                       d_timeIntegrator(timeIntegrator) {};

  virtual ~DQMOMEqnBuilderBase(){};

  virtual EqnBase* build() = 0;  

protected: 
  ArchesLabel* d_fieldLabels; 
  string d_eqnName; 
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
class DQMOMEqnFactory
{
public:

  typedef std::map< std::string, EqnBase* >     EqnMap; 

  /** @brief Return an instance of the factory.  */
  static DQMOMEqnFactory& self(); 

  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            DQMOMEqnBuilderBase* builder);

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

}; // class DQMOMEqnFactory 
} // end namespace Uintah

#endif
