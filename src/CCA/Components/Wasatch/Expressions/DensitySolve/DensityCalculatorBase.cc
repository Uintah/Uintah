#include <CCA/Components/Wasatch/Expressions/DensitySolve/DensityCalculatorBase.h>
#include <sci_defs/uintah_defs.h>

namespace WasatchCore{
namespace DelMe{

  DensityCalculatorBase::
  DensityCalculatorBase( const std::string treeName,
                         const Expr::TagList phiTags,
                         const double rTol,
                         const size_t maxIter )
  : setupHasRun_(false),
    treeName_(treeName),
    rhoPhiTags_  (tags_with_prefix(phiTags, "solver_rho"     )),
    phiOldTags_  (tags_with_prefix(phiTags, "solver_old"     )),
    phiNewTags_  (tags_with_prefix(phiTags, "solver_new"     )),
    residualTags_(tags_with_prefix(phiTags, "solver_residual")),
    densityTag_("solve_density", Expr::STATE_NONE),
    rTol_(rTol),
    maxIter_(maxIter)
    {}

  //-------------------------------------------------------------------

  void
  DensityCalculatorBase::
  bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    patchContainer_ = opDB.retrieve_operator<UintahPatchContainer>();
  }

  //-------------------------------------------------------------------

  void
  DensityCalculatorBase::
  setup()
  {
    using namespace SpatialOps;
    std::cout << "\ncalling DensityCalculatorBase::setup()...";

    const Uintah::Patch* patch = patchContainer_->get_uintah_patch();

    helper_.set_alloc_info(patch);
    std::cout << "\ncalling register_local_expressions()...";
    Expr::IDSet rootIDs = register_local_expressions();
    newtonSolveTreePtr_ = helper_.new_tree(treeName_, rootIDs);

    helper_.finalize();
    setupHasRun_ = true;
    std::cout << "done \n";
  }

  //-------------------------------------------------------------------
  DensityCalculatorBase::
  ~DensityCalculatorBase()
  {}
  //-------------------------------------------------------------------

}
}