/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef WasatchDensityCalculatorBase_h
#define WasatchDensityCalculatorBase_h

#include <expression/Expression.h>
#include <expression/ManagerTypes.h>

#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/OperatorDatabase.h>

#include <CCA/Components/Wasatch/NestedGraphHelper.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <spatialops/structured/FieldHelper.h>

namespace WasatchCore{
namespace DelMe{

  template< typename FieldT >
  class DensityCalculatorBase : public Expr::Expression<FieldT>
  {
    public:
       /**
       *  @brief Base class for expressions that compute density for Wasatch's low-Mach NSE solver
       *  @param densityTag tag to the density which will be calculated
       *  @param phiTags tags to primitive (transported) scalars 
       *  @param rTol relative tolerance
       *  @param maxIter maximum number of iterations for Newton solve used to compute density
       */
      DensityCalculatorBase( const Expr::Tag densityTag,
                             const Expr::TagList dRhodPhiTags,
                             const Expr::TagList phiTags,
                             const double rTol,
                             const size_t maxIter=20 )
      : Expr::Expression<FieldT>(),
        setupHasRun_(false),
        densityOldTag_(tag_with_prefix (densityTag  , "solver_old"     )),
        densityNewTag_(tag_with_prefix (densityTag  , "solver_new"     )),
        dRhodPhiTags_ (tags_with_prefix(dRhodPhiTags, "solver_old"     )),
        rhoPhiTags_   (tags_with_prefix(phiTags     , "solver_rho"     )),
        phiOldTags_   (tags_with_prefix(phiTags     , "solver_old"     )),
        phiNewTags_   (tags_with_prefix(phiTags     , "solver_new"     )),
        residualTags_ (tags_with_prefix(phiTags     , "solver_residual")),
        rTol_(rTol),
        maxIter_(maxIter)
        {};

    //-------------------------------------------------------------------

    protected:
      bool                  setupHasRun_;
      const Expr::Tag       densityOldTag_, densityNewTag_;
      const Expr::TagList   dRhodPhiTags_, rhoPhiTags_, phiOldTags_, phiNewTags_, residualTags_;
      const double          rTol_;
      const unsigned        maxIter_;
      NestedGraphHelper     helper_;
      UintahPatchContainer* patchContainer_;
      Expr::ExpressionTree *newtonSolveTreePtr_, *dRhodPhiTreePtr_;

      virtual Expr::IDSet register_local_expressions() = 0;
      virtual void set_initial_guesses() = 0;

      void bind_operators( const SpatialOps::OperatorDatabase& opDB )
      {
        patchContainer_ = opDB.retrieve_operator<UintahPatchContainer>();
      }

      //-------------------------------------------------------------------

      void setup()
      {
        using namespace SpatialOps;
        std::cout << "\ncalling DensityCalculatorBase::setup()...";
    
        const Uintah::Patch* patch = patchContainer_->get_uintah_patch();
    
        helper_.set_alloc_info(patch);
        std::cout << "\ncalling register_local_expressions()...";
        const Expr::IDSet newtonSolveIDs = register_local_expressions();

        Expr::IDSet dRhodPhiIDs;
        for(const Expr::Tag& tag : dRhodPhiTags_)
        {
          dRhodPhiIDs.insert(helper_.factory_->get_id(tag));
        }

        newtonSolveTreePtr_ = helper_.new_tree("density_newton_iteration", newtonSolveIDs);
        dRhodPhiTreePtr_    = helper_.new_tree("d_rho_d_phi_eval"        , dRhodPhiIDs   );

        helper_.finalize();
        setupHasRun_ = true;
        std::cout << "done \n";
      }

      //-------------------------------------------------------------------

      static Expr::Tag tag_with_prefix( const Expr::Tag& tag,
                                        const std::string prefix )
      {                                   
        return Expr::Tag(prefix + "_" + tag.name(), Expr::STATE_NONE);
      }

      //-------------------------------------------------------------------

      static Expr::TagList tags_with_prefix( const Expr::TagList& tags,
                                             const std::string prefix )
      {
        Expr::TagList resTags;
        for (const Expr::Tag& tag : tags){
          resTags.push_back( tag_with_prefix(tag, prefix) );
        }
        return resTags;
      }

    // virtual ~DensityCalculatorBase(){};
  };

}
}

#endif // WasatchDensityCalculatorBase_h
