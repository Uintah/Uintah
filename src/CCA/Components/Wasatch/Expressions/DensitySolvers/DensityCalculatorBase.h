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
                             const Expr::TagList betaTags )
      : Expr::Expression<FieldT>(),
        setupHasRun_  (false           ),
        nEq_          ( phiTags.size() ),
        delta_        ( 1e-3           ),
        densityOldTag_(tag_with_prefix (densityTag  , "solver_old"     ) ),
        densityNewTag_(tag_with_prefix (densityTag  , "solver_new"     ) ),
        rhoPhiTags_   (tags_with_prefix(phiTags     , "solver_rho"     ) ),
        phiOldTags_   (tags_with_prefix(phiTags     , "solver_old"     ) ),
        phiNewTags_   (tags_with_prefix(phiTags     , "solver_new"     ) ),
        betaOldTags_  (tags_with_prefix(betaTags    , "solver_old"     ) ),
        betaNewTags_  (tags_with_prefix(betaTags    , "solver_new"     ) ),
        residualTags_ (tags_with_prefix(phiTags     , "solver_residual") ),
        dRhodPhiTags_ (density_derivative_tags_with_prefix(phiTags)      ),
        dRhodBetaTags_(density_derivative_tags_with_prefix(betaTags)     ),
        rTol_   ( rTol    ),
        maxIter_( maxIter )
        {
          for(unsigned i = 0; i < nEq_; ++i){
            if(phiNewTags_[i] != betaNewTags_[i]){
              phiUpdateIndecices_.push_back(i);
            }
          }
        };

    //-------------------------------------------------------------------

    protected:
      bool                  setupHasRun_;
      const unsigned        nEq_;
      const double          delta_; // prevents relative error numerator from becoming zero
      const Expr::Tag       densityOldTag_, densityNewTag_;
      const Expr::TagList   rhoPhiTags_, phiOldTags_, phiNewTags_, betaOldTags_, betaNewTags_,
                            residualTags_, dRhodPhiTags_, dRhodBetaTags_;
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
    
        const Uintah::Patch* patch = patchContainer_->get_uintah_patch();
        helper_.set_alloc_info(patch, this->is_gpu_runnable());

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
      }

      

    //-------------------------------------------------------------------

      double compute_error( Expr::UintahFieldManager<FieldT>& fm,
                            FieldT& error, 
                            Expr::Tag& worstFieldTag )
      {
        double maxError = 0;
        for(unsigned i=0; i<this->nEq_; i++){

          const FieldT& betaOld = fm.field_ref( this->betaOldTags_[i] );
          const FieldT& betaNew = fm.field_ref( this->betaNewTags_[i] );

          error <<= abs(betaNew - betaOld);

          const double betaError = nebo_max(error)/get_normalization_factor(i);

          if(betaError > maxError){
            maxError = betaError;
            worstFieldTag = this->betaNewTags_[i];
          }
        }
        return maxError;
      }

    //-------------------------------------------------------------------

      double newton_solve()
      {
        using namespace SpatialOps;
        // setup() needs to be run here because we need fields to be defined before a local patch can be created
        if( !this->setupHasRun_ ){ this->setup();}

        set_initial_guesses();

        Expr::FieldManagerList* fml = this->helper_.fml_;

        Expr::ExpressionTree& newtonSolveTree = *(this->newtonSolveTreePtr_);
        newtonSolveTree.bind_fields( *fml );
        newtonSolveTree.lock_fields( *fml ); // this is needed... why?

        Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

        unsigned numIter = 0;
        bool converged = false;

        double maxError = 0;

        // get a scratch field for error computation
        SpatFldPtr<FieldT> error = SpatialFieldStore::get<FieldT>( fieldTManager.field_ref( this->densityOldTag_ ) );

        Expr::Tag badTag;
        while(numIter< this->maxIter_ && !converged)
        {
          ++numIter;
          newtonSolveTree.execute_tree();

          maxError = compute_error(fieldTManager, *error, badTag);
          converged = (maxError <= this->rTol_);

          for(unsigned i=0; i<this->nEq_; i++){
            FieldT& betaOld = fieldTManager.field_ref( this->betaOldTags_[i] );
            betaOld <<= fieldTManager.field_ref( this->betaNewTags_[i] );
          }
          
          for(const unsigned& i : phiUpdateIndecices_){
            FieldT&       phiOld  = fieldTManager.field_ref( this->phiOldTags_[i] );
            phiOld <<= fieldTManager.field_ref( this->phiNewTags_[i] );
          }

          // update variables for next iteration and check if error is below tolerance
          FieldT& rhoOld = fieldTManager.field_ref( this->densityOldTag_ );
          rhoOld <<= fieldTManager.field_ref( this->densityNewTag_ );
        }

        if(!converged){
          std::cout << "\tSolve for density FAILED (max error = " << maxError << " field: "<< badTag.name() << ") after " 
                    << numIter << " iterations.\n";
        }
        #ifndef NDEBUG
        else{
          std::cout << "\tSolve for density completed (max error = " << maxError << ") after " << numIter << " iterations.\n";
        }
        #endif


        Expr::ExpressionTree& dRhodFTree = *(this->dRhodPhiTreePtr_);
        dRhodFTree.bind_fields( *fml );
        dRhodFTree.lock_fields( *fml );
        dRhodFTree.execute_tree();

        return maxError;
      }

      //-------------------------------------------------------------------

      void unlock_fields()
      {
        Expr::FieldManagerList* fml = this->helper_.fml_;
        this->newtonSolveTreePtr_->unlock_fields( *fml );
        this->dRhodPhiTreePtr_   ->unlock_fields( *fml );
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
          newTags.push_back( Expr::Tag("solver_d_rho_d_" + tag.name(), Expr::STATE_NONE) );
        }
        return newTags;
      }

     //-------------------------------------------------------------------

     virtual double get_normalization_factor(const unsigned i) const =0;

    private:
      // These are indecies of transported scalars that are computed from betaValues
      // and not from the Newton update directly, i.e., h = h(T, Y) where h is 
      // enthalpy, T is temperature and Y is composition. Because of this, computed  
      // values of these scalars need to be copied to old values in order to  compute
      // all residuals required for the Newton solve.
      std::vector<unsigned> phiUpdateIndecices_;
  };
}

#endif // WasatchDensityCalculatorBase_h
