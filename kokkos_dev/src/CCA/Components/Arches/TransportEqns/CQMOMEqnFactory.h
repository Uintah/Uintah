#ifndef UT_CQMOMEqnFactory_h
#define UT_CQMOMEqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h> // should this be here?
#include <Core/Grid/Variables/VarLabel.h>
#include <map>
#include <vector>
#include <string>

//---------------------------------------------------------------------------
// Builder

/**
 * @class CQMOMEqnBuilder
 * @author Alex Abboud, adaption from DQMOMEqnFactory
 * @date May 2014
 *
 * @brief Abstract base class to support scalar equation additions.  Meant to
 * be used with the CQMOMEqnFactory.
 *
 */
namespace Uintah {
  class EqnBase;
  class CQMOMEqnBuilderBase
  {
  public:
    CQMOMEqnBuilderBase( ArchesLabel* fieldLabels,
                        ExplicitTimeInt* timeIntegrator,
                        std::string eqnName ) :
    d_fieldLabels(fieldLabels),
    d_eqnName(eqnName),
    d_timeIntegrator(timeIntegrator) {};
    
    virtual ~CQMOMEqnBuilderBase(){};
    
    virtual EqnBase* build() = 0;
    
  protected:
    ArchesLabel* d_fieldLabels;
    std::string d_eqnName;
    ExplicitTimeInt* d_timeIntegrator;
  }; // class CQMOMEqnBuilder
  
  // End builder
  //---------------------------------------------------------------------------
  
  /**
   * @class  CQMOMEqnFactory
   * @author Alex Abboud, adaptino from DQMOMEQnFactory
   * @date   May 2014
   *
   * @brief  A Factory for building moments eqns for CQMOM.
   *
   */
  class CQMOMEqnFactory
  {
  public:
    
    typedef std::map< std::string, EqnBase* > EqnMap;
    
    /** @brief Return an instance of the factory.  */
    static CQMOMEqnFactory& self();
    
    /** @brief Register a scalar eqn with the builder.    */
    void register_scalar_eqn( const std::string name,
                             CQMOMEqnBuilderBase* builder);
    
    /** @brief register an equation for a moment */
    void set_moment_eqn( const std::string name, EqnBase* eqn);
    
    /** @brief Retrieve a given scalar eqn.    */
    EqnBase& retrieve_scalar_eqn( const std::string name );
    
    /** @brief Determine if a given scalar eqn is contained in the factory. */
    bool find_scalar_eqn( const std::string name );
    
    /** @brief Get access to the eqn map */
    EqnMap& retrieve_all_eqns(){ return eqns_; };
    
    /** @brief Get access to the moments eqn map */
    EqnMap& retrieve_moments_eqns(){ return moments_eqns; };
    
    /** @brief Set number moment eqns */
    inline void set_number_moments( int nM ) { nMoments = nM; };
    
    /** @brief Get number of moment eqns */
    inline int get_number_moments() { return nMoments; };
    
  private:
    
    typedef std::map< std::string, CQMOMEqnBuilderBase* >  BuildMap;
    
    BuildMap builders_;
    EqnMap eqns_;
    EqnMap moments_eqns;
    int nMoments;
    
    CQMOMEqnFactory();
    ~CQMOMEqnFactory();
    
    bool doing_CQMOM_;
    
  }; // class CQMOMEqnFactory 
} // end namespace Uintah

#endif
