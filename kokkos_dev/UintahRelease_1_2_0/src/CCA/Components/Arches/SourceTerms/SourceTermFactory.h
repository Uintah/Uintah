#ifndef UT_SourceTermFactory_h
#define UT_SourceTermFactory_h

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>
#include <map>
#include <vector>
#include <string>

//====================================================================

/**
 *  @class  SourceTermBuilder
 *  @author James C. Sutherland and Jeremy Thornock
 *  @date   November, 2006
 *
 *  @brief Abstract base class to support source term
 *  additions. Should be used in conjunction with the
 *  SourceTermFactory.
 *
 *  An arbitrary number of source terms may be added to a transport
 *  equation via the SourceTermFactory.  The SourceTermBuilder object
 *  is passed to the factory to provide a mechanism to instantiate the
 *  SourceTerm object.
 */
namespace Uintah {
//---------------------------------------------------------------------------
// Builder
class SourceTermBase; 
class SourceTermBuilder
{
public:
  SourceTermBuilder(std::string src_name, vector<std::string> reqLabelNames, 
                    SimulationStateP& sharedState) : 
                    d_srcName(src_name), d_requiredLabels(reqLabelNames), 
                    d_sharedState(sharedState){};
  virtual ~SourceTermBuilder(){};

  /**
   *  build the SourceTerm.  Should be implemented using the
   *  "scinew" operator.  Ownership is transfered.
   */
  virtual SourceTermBase* build() = 0;

protected: 
  std::string d_srcName;
  vector<string> d_requiredLabels; 
  SimulationStateP& d_sharedState; 

private: 
};
// End Builder
//---------------------------------------------------------------------------

/**
 *  @class  SourceTermFactory
 *  @author James C. Sutherland and Jeremy Thornock
 *  @date   November, 2006
 *  @brief  Factory for source term generation.
 *
 *  Allows easy addition of source terms to a transport equation.
 *  Simply register the builder object for your SourceTerm with
 *  the factory and it will automatically be added to the requested
 *  TransportEquation.  Multiple source terms may be registered with a
 *  single transport equation.
 *
 *  Implemented as a singleton.
 */
class SourceTermFactory
{
public:
  /**
   *  @brief obtain a reference to the SourceTermFactory.
   */
  static SourceTermFactory& self();

  /**
   *  @brief Register a source term on the specified transport equation.
   *
   *  @param eqnName The name of the transport equation to place the source term on.
   *  @param builder The SourceTermBuilder object to build the SourceTerm object.
   *
   *  SourceTermBuilder objects should be heap-allocated using "new".
   *  Memory management will be transfered to the SourceTermFactory.
   */
  void register_source_term( const std::string name,
                             SourceTermBuilder* builder );


  /**
   *  @brief Retrieve a vector of pointers to all SourceTerm
   *  objects that have been assigned to the transport equation with
   *  the specified name.
   *
   *  @param eqnName The name of the transport equation to retrieve
   *  SourceTerm objects for.
   *
   *  Note that this will construct new objects only as needed.
   */
  SourceTermBase& retrieve_source_term( const std::string name );

  typedef std::map< std::string, SourceTermBuilder* > BuildMap;
  typedef std::map< std::string, SourceTermBase*        > SourceMap;

  // get all source terms
  SourceMap& retrieve_all_sources(){
    return sources_; }; 

private:

  BuildMap builders_;
  SourceMap sources_;

  SourceTermFactory();
  ~SourceTermFactory();
}; // class SourceTermFactory
}  //Namespace Uintah
#endif
