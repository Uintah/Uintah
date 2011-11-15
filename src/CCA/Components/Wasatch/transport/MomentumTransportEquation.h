#ifndef Wasatch_MomentumTransportEquation_h
#define Wasatch_MomentumTransportEquation_h

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SolverInterface.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \class MomentumTransportEquation
   *  \author James C. Sutherland
   *  \author Tony Saad
   *  \date January, 2011
   *
   *  \brief Creates a momentum transport equation
   *
   *  \todo Allow more flexibility in specifying initial and boundary conditions for momentum.
   */
  template< typename FieldT >
  class MomentumTransportEquation : public Wasatch::TransportEquation
  {
  public:

    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a MomentumTransportEquation
     *  \param velName the name of the velocity component solved by this MomentumTransportEquation
     *  \param momName the name of the momentum component solved by this MomentumTransportEquation
     *  \param factory the Expr::ExpressionFactory that will hold expressions registered by this transport equation.
     *  \param params Parser information for this momentum equation
     *  \param linSolver the linear solver object for the pressure solve
     */
    MomentumTransportEquation( const std::string velName,
                               const std::string momName,
                               const Expr::Tag densTag,
                               Expr::ExpressionFactory& factory,
                               Uintah::ProblemSpecP params,
                               Uintah::SolverInterface& linSolver);

    ~MomentumTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this momentum equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials);
    /**
     *  \brief setup the initial conditions for this momentum equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     *  \brief Parse the input file to get the name of this MomentumTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_phi_name( Uintah::ProblemSpecP params );

  private:

    const bool isviscous_;
    Expr::ExpressionID normalStressID_, normalConvFluxID_, pressureID_;
    std::string thisMomName_;
    Expr::TagList velTags_; ///< TagList for the velocity expressions

  };

} // namespace Wasatch

#endif // Wasatch_MomentumTransportEquation_h
