#ifndef UT_EqnFactory_h
#define UT_EqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <map>
#include <vector>
#include <string>

//---------------------------------------------------------------------------
// Builder 

/**
  * @class EqnBuilder
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date November 19, 2008
  *
  * @brief Abstract base class to support scalar equation additions.  Meant to
  * be used with the EqnFactory. 
  *
  */
namespace Uintah {
class EqnBase; 
class EqnBuilder
{
public:
  EqnBuilder( ArchesLabel* fieldLabels, 
              ExplicitTimeInt* timeIntegrator,
              string eqnName ) : 
              d_fieldLabels(fieldLabels), 
              d_timeIntegrator(timeIntegrator),
              d_eqnName(eqnName) {};
  virtual ~EqnBuilder(){};

  virtual EqnBase* build() = 0;  

protected: 
  ArchesLabel* d_fieldLabels; 
  ExplicitTimeInt* d_timeIntegrator;
  string d_eqnName; 
}; // class EqnBuilder

// End builder 
//---------------------------------------------------------------------------

/**
  * @class  EqnFactory
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date   November 19, 2008
  * 
  * @brief  A Factory for building eqns. 
  * 
  */
class EqnFactory
{
public:

  typedef std::map< std::string, EqnBase* >     EqnMap; 

  /** @brief Return an instance of the factory.  */
  static EqnFactory& self(); 
  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            EqnBuilder* builder);
  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );

  /** @brief Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };
private:
  typedef std::map< std::string, EqnBuilder* >  BuildMap; 

  BuildMap builders_; 
  EqnMap eqns_; 

  EqnFactory(); 
  ~EqnFactory(); 
}; // class EqnFactory 
} // end namespace Uintah

#endif
