#ifndef Wasatch_MomentTransportEquation_h
#define Wasatch_MomentTransportEquation_h

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Wasatch{
  
  /**
   *  \ingroup WasatchCore
   *  \class MomentTransportEquation
   *  \date April, 2011
   *  \author Tony Saad
   *
   *  \brief Moment transport equation for population balances with a single
   *				 internal coordinate.
   *
   *  Sets up solution for a transport equation of the form:
   *
   *  \f[
   *    \frac{\partial m_k}{\partial t} =
   *    - \frac{\partial m_k u_x }{\partial x} 
   *    - \frac{\partial m_k u_y }{\partial y} 
   *    - \frac{\partial m_k u_z }{\partial z} 
   *    - \frac{\partial J_{m_k,x}}{\partial x}
   *    - \frac{\partial J_{m_k,y}}{\partial y}
   *    - \frac{\partial J_{m_k,z}}{\partial z}
   *    + s_\phi
   *  \f]
   *
   *  Any or all of the terms in the RHS above may be activated
   *  through the input file.
   *
   */
  template<typename FieldT>
  class MomentTransportEquation : public Wasatch::TransportEquation
  {
  public:        
    /**
     *  \brief Construct a MomentTransportEquation
     *  \param basePhiName This equation will be created n-times where n is a user
     *         specified number in the input file. The basePhiName refers to the
     *         base name of the solution variable. The n-equations that are created
     *         will correspond to basePhiName0, basePhiName1, etc...
     *  \param thisPhiName The name of the solution variable for this ScalarTransportEquation
     *  \param id The Expr::ExpressionID for the RHS expression for this ScalarTransportEquation
     *
     *  Note that the static member method get_rhs_expr_id can be useful to 
     *  obtain the appropriate input arguments here.
     */
    MomentTransportEquation(const std::string thisPhiName,
                            const Expr::ExpressionID id );
    
    ~MomentTransportEquation();
    
    
    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                   const Uintah::PatchSet* const localPatches,
                                   const PatchInfoMap& patchInfoMap,
                                   const Uintah::MaterialSubset* const materials);
    
    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );
    
    /**
     * \brief Parse the input file to determine the rhs expression id.
     *        Also registers convective flux, diffusive flux, and
     *        source term expressions.
     *
     *  \param factory The Expr::ExpressionFactory object that
     *         terms associated with the RHS of this transport
     *         equation should be registered on.
     *
     *  \param params The Uintah::ProblemSpec XML description for this
     *         equation.  Scope should be within the ScalabilityTest tag.
     */
    static Expr::ExpressionID  get_moment_rhs_id(Expr::ExpressionFactory& factory,
                                                  Uintah::ProblemSpecP params,
                                                  Expr::TagList& weightsTags,
                                                  Expr::TagList& abscissaeTags,                    
                                                  const double momentOrder);    
    
    /**
     *  \brief Parse the input file to get the name of this ScalarTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    //static std::string get_phi_name( Uintah::ProblemSpecP params );
    
  private:
    //Expr::TagList weightsTagList_, abscissaeTagList_, weightsAndAbscissaeTagList_;
    };

  template< typename FieldT >
  void setup_growth_expression(	Uintah::ProblemSpecP growthParams,
                               const std::string& phiName, 
                               const double momentOrder,
                               const int nEqs,
                               Expr::TagList& growthTags,
                               const Expr::TagList& weightsTagList,
                               const Expr::TagList& abscissaeTagList,                               
                               Expr::ExpressionFactory& factory );
  
  template< typename FieldT >
  void setup_nucleation_expression(Uintah::ProblemSpecP nucleationParams,
                                   const std::string& phiName, 
                                   Expr::ExpressionFactory& factory,
                                   typename MomentRHS<FieldT>::FieldTagInfo& info );  

  template< typename FieldT >
  void setup_birth_expression( Uintah::ProblemSpecP birthParams,
                              const std::string& phiName,
															Expr::ExpressionFactory& factory,
                              typename MomentRHS<FieldT>::FieldTagInfo& info );  

  template< typename FieldT >
  void setup_death_expression(Uintah::ProblemSpecP deathParams,
                              const std::string& phiName, 
                              Expr::ExpressionFactory& factory,
                              typename MomentRHS<FieldT>::FieldTagInfo& info );  

  template< typename FieldT >
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                       const std::string& phiName,
                                       Expr::ExpressionFactory& factory,
                                       typename MomentRHS<FieldT>::FieldTagInfo& info );

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                        const std::string& phiName,
                                        Expr::ExpressionFactory& factory,
                                        typename MomentRHS<FieldT>::FieldTagInfo& info );  
  
  std::string
  get_population_name( Uintah::ProblemSpecP params );

} // namespace Wasatch
#endif // Wasatch_MomentTransportEquation_h


