#ifndef UT_EqnFactory_h
#define UT_EqnFactory_h

#include <CCA/Components/SpatialOps/Fields.h>
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
  EqnBuilder( const Fields* fieldLabels, 
              const VarLabel* transportVarLabel, 
              string eqnName ) : 
              d_fieldLabels(fieldLabels), 
              d_transportVarLabel(transportVarLabel), 
              d_eqnName(eqnName) {};
  virtual ~EqnBuilder(){};

  virtual EqnBase* build() = 0;  

protected: 
  const Fields* d_fieldLabels; 
  const VarLabel* d_transportVarLabel; 
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
  /** @brief Return an instance of the factory.  */
  static EqnFactory& self(); 
  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            EqnBuilder* builder);
  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

private:
  typedef std::map< std::string, EqnBuilder* >  BuildMap; 
  typedef std::map< std::string, EqnBase* >     EqnMap; 

  BuildMap builders_; 
  EqnMap eqns_; 

  EqnFactory(); 
  ~EqnFactory(); 
}; // class EqnFactory 
} // end namespace Uintah

#endif
