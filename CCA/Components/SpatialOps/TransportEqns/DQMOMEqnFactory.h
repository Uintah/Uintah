#ifndef UT_DQMOMEqnFactory_h
#define UT_DQMOMEqnFactory_h

#include <CCA/Components/SpatialOps/Fields.h>
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
  DQMOMEqnBuilderBase( const Fields* fieldLabels, 
              const VarLabel* transportVarLabel, 
              string eqnName ) : 
              d_fieldLabels(fieldLabels), 
              d_transportVarLabel(transportVarLabel), 
              d_eqnName(eqnName) {};
  virtual ~DQMOMEqnBuilderBase(){};

  virtual EqnBase* build() = 0;  

protected: 
  const Fields* d_fieldLabels; 
  const VarLabel* d_transportVarLabel; 
  string d_eqnName; 
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
  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /** @brief Set number quadrature nodes */ 
  inline void set_quad_nodes( int qn ) {
    n_quad_ = qn; };

  /** @brief Set the number of quadrature nodes */ 
  inline const int get_quad_nodes( ) {
    return n_quad_; };

private:
  typedef std::map< std::string, DQMOMEqnBuilderBase* >  BuildMap; 

  BuildMap builders_; 
  EqnMap eqns_; 

  DQMOMEqnFactory(); 
  ~DQMOMEqnFactory(); 

  int n_quad_; 

}; // class DQMOMEqnFactory 
} // end namespace Uintah

#endif
