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

  /**
   * \class DensityCalculatorBase
   *  \author Josh McConnell
   *  \date January 2020
   *
   * Base class which provides basic structure for expressions that calculate density at time level 
   * \f$n+1\f$  (\f$\rho^{n+1}\f$)  from scalars transported in strong form (\f$(\rho\phi)_i^{n+1}\f$), 
   * and a set of variables at time level \f$n\f$ (\f$\beta_j^{n}\f$) where primitive scalars \f$\phi_i\f$ 
   * are functions of variables \f$\beta_j\f$, \emph{i.e.} \f$\phi_i = \phi_i(\beta_j) \f$.
   * 
   * The procedure for calculating the density is as follows:
   * \begin{enumerate}
   *  \item compute \f$\phi_i\f$ and \f$\rho\f$ from guesses for \f$\beta_j\f$
   *  \item compute residuals \f$r_i = \rho_i\phi_i - (\rho\phi)_i^{n+1} \f$
   *  \item compute residual Jacobian matrix elements \f$[J]_{ij} = \frac{\partial r_i}{\partial \beta_j}\f$
   *  \item solve the linearized system \f$[J] \vec{\Delta} = \vec{r}\f$
   *  \item update guesses for varialbles \f$\beta_j = \beta_j -\Delta_j\f$
   * \end{enumerate}
   * 
   * This procedure ia repeated until the resiudals are below a specified tolerance. Note, in some cases, 
   *  \f$\phi_i = \beta_i\f$, which somewhat simplifies the above procedure.
   *  
   */

  template< typename FieldT >
  class DensityCalculatorBase : public Expr::Expression<FieldT>
  {
    public:
       /**
       *  @brief Base class for expressions that compute density for Wasatch's low-Mach NSE solver
       *  @param densityTag tag to the density, \f$\rho = \rho(\beta_j)\f$ which will be calculated 
       *  @param phiTags tags to primitive transported scalars \f$\phi_i = \phi_i(\beta_j) \f$
       *  @param phiTags tags to primitive transported scalars \f$\phi_i = \phi_i(\beta_j) \f$
       *  @param rTol relative tolerance
       *  @param maxIter maximum number of iterations for Newton solve used to compute density
       */
      DensityCalculatorBase( const double rTol,
                             const size_t maxIter,
                             const Expr::Tag densityTag,
                             const Expr::TagList phiTags,
                             const Expr::TagList betaTags = Expr::TagList() )
      : Expr::Expression<FieldT>(),
        setupHasRun_(false),
        densityOldTag_(tag_with_prefix (densityTag  , "solver_old"     )),
        densityNewTag_(tag_with_prefix (densityTag  , "solver_new"     )),
        rhoPhiTags_   (tags_with_prefix(phiTags     , "solver_rho"     )),
        phiOldTags_   (tags_with_prefix(phiTags     , "solver_old"     )),
        phiNewTags_   (tags_with_prefix(phiTags     , "solver_new"     )),
        betaOldTags_  (tags_with_prefix(betaTags    , "solver_old"     )),
        betaNewTags_  (tags_with_prefix(betaTags    , "solver_new"     )),
        residualTags_ (tags_with_prefix(phiTags     , "solver_residual")),
        dRhodPhiTags_ (density_derivative_tags_with_prefix(phiTags)     ),
        dRhodBetaTags_(density_derivative_tags_with_prefix(betaTags)    ),
        rTol_(rTol),
        maxIter_(maxIter)
        {};

    //-------------------------------------------------------------------

    protected:
      bool                  setupHasRun_;
      const Expr::Tag       densityOldTag_, densityNewTag_;
      const Expr::TagList   rhoPhiTags_, phiOldTags_, phiNewTags_, betaOldTags_, betaNewTags_,
                            residualTags_, dRhodPhiTags_, dRhodBetaTags;
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

      // \brief returns an Expr::Tag with a prefix preppended to the tag name 
      static Expr::Tag tag_with_prefix( const Expr::Tag& tag,
                                        const std::string prefix )
      {                                   
        return Expr::Tag(prefix + "_" + tag.name(), Expr::STATE_NONE);
      }

      //-------------------------------------------------------------------

      // \brief returns an Expr::TagList with a prefix preppended to the tag names 
      static Expr::TagList tags_with_prefix( const Expr::TagList& tags,
                                             const std::string prefix )
      {
        Expr::TagList newTags;
        for (const Expr::Tag& tag : tags){
          newTags.push_back( tag_with_prefix(tag, prefix) );
        }
        return newTags;
      }

      //-------------------------------------------------------------------
      
      // \brief returns an Expr::TagList for density derivatives 
      static Expr::TagList density_derivative_tags_with_prefix( const Expr::TagList& tags )
      {
        Expr::TagList newTags;
        for (const Expr::Tag& tag : tags){
          newTags.push_back( "solver_d_rho_d_" + tag.name(), Expr::STATE_NONE );
        }
        return newTags;
      }

    // virtual ~DensityCalculatorBase(){};
  };

}

#endif // WasatchDensityCalculatorBase_h
