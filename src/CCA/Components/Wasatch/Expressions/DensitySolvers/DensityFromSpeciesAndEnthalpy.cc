/*
 * The MIT License
 *
 * Copyright (c) 2012-2023 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/DensitySolvers/DensityFromSpeciesAndEnthalpy.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/Residual.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/NewtonUpdate.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/QuotientFunction.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/DEnthalpyDY.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/DRhoDEnthalpy.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/DRhoDTemperature.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/SpeciesAndEnthalpyExpressions/DRhoDY.h>
#include <CCA/Components/Wasatch/Expressions/DensitySolvers/HelperFunctions.h>

#include <pokitt/CanteraObjects.h>
#include <pokitt/SpeciesN.h>
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/thermo/Enthalpy.h>
#include <pokitt/thermo/Temperature.h>
#include <pokitt/thermo/HeatCapacity_Cp.h>
#include <pokitt/thermo/Density.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <expression/ExprLib.h>
#include <expression/Expression.h>
#include <expression/ExpressionFactory.h>
#include <expression/ClipValue.h>

#include <expression/matrix-assembly/MapUtilities.h>
#include <expression/matrix-assembly/SparseMatrix.h>
#include <expression/matrix-assembly/ScaledIdentityMatrix.h>

#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FieldHelper.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/MatVecOps.h>

#include <sci_defs/uintah_defs.h>


namespace WasatchCore{

  using Expr::tag_list;
  typedef std::pair<double, double> BoundsT;

  template< typename FieldT >
  DensityFromSpeciesAndEnthalpy<FieldT>::
  DensityFromSpeciesAndEnthalpy( const Expr::Tag&     rhoOldTag,
                                 const Expr::TagList& rhoYTags,
                                 const Expr::Tag&     rhoHTag,
                                 const Expr::TagList& yOldTags,
                                 const Expr::Tag&     hOldTag,
                                 const Expr::Tag&     temperatureOldTag,
                                 const Expr::Tag&     mmwOldTag,
                                 const Expr::Tag&     pressureTag,
                                 const double         rTol,
                                 const unsigned       maxIter )
    : DensityCalculatorBase<FieldT>( rTol, 
                                     maxIter,
                                     rhoOldTag, 
                                     tag_list(sub_vector(yOldTags,0, yOldTags.size()-1), hOldTag          ),
                                     tag_list(sub_vector(yOldTags,0, yOldTags.size()-1), temperatureOldTag) ),
      nSpec_             ( CanteraObjects::number_species() ),
      yOldTags_          ( tag_list(sub_vector(this->betaOldTags_  , 0, this->nEq_-1), 
                                    this->tag_with_prefix(yOldTags[nSpec_-1], "solver_old") )),
      yNewTags_          ( tag_list(sub_vector(this->betaNewTags_  , 0, this->nEq_-1), 
                                    this->tag_with_prefix(yOldTags[nSpec_-1], "solver_new") )),
      rhoYTags_          ( sub_vector(this->rhoPhiTags_   , 0, this->nEq_-1) ),
      dRhodYTags_        ( sub_vector(this->dRhodBetaTags_, 0, this->nEq_-1) ),
      hOldTag_           ( this->phiOldTags_   [this->nEq_-1] ),
      hNewTag_           ( this->phiNewTags_   [this->nEq_-1] ),
      rhoHTag_           ( this->rhoPhiTags_   [this->nEq_-1] ),
      dRhodHTag_         ( this->dRhodPhiTags_ [this->nEq_-1] ),
      temperatureOldTag_ ( this->betaOldTags_  [this->nEq_-1] ),
      temperatureNewTag_ ( this->betaNewTags_  [this->nEq_-1] ),
      dRhodTempertureTag_( this->dRhodBetaTags_[this->nEq_-1] ),
      pressureTag_       ("solver_"+pressureTag.name(), Expr::STATE_NONE),
      mmwOldTag_         ( this->tag_with_prefix(mmwOldTag, "solver_old")),
      mmwNewTag_         ( this->tag_with_prefix(mmwOldTag, "solver_new")),
      cpTag_             ( "solver_cp", Expr::STATE_NONE)
  {
    assert(this->phiOldTags_  .size() == nSpec_);
    assert(this->phiNewTags_  .size() == nSpec_);
    assert(this->betaOldTags_ .size() == nSpec_);
    assert(this->betaNewTags_ .size() == nSpec_);
    assert(this->residualTags_.size() == nSpec_);

    this->set_gpu_runnable(true);
    this->template create_field_vector_request<FieldT>( rhoYTags, rhoY_ );
    this->template create_field_vector_request<FieldT>( yOldTags, yOld_ );

    rhoH_           = this->template create_field_request<FieldT>( rhoHTag           );
    rhoOld_         = this->template create_field_request<FieldT>( rhoOldTag         );
    hOld_           = this->template create_field_request<FieldT>( hOldTag           );
    temperatureOld_ = this->template create_field_request<FieldT>( temperatureOldTag );
    mmwOld_         = this->template create_field_request<FieldT>( mmwOldTag         );
    pressure_       = this->template create_field_request<FieldT>( pressureTag       );

    hiTags_.clear();
    for(int i=0; i<nSpec_; ++i){
      const std::string specName = CanteraObjects::species_name(i);
      hiTags_.push_back(Expr::Tag("solver_h_" + specName, Expr::STATE_NONE));
    }
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromSpeciesAndEnthalpy<FieldT>::
  ~DensityFromSpeciesAndEnthalpy()
  {}

  //--------------------------------------------------------------------

  template< typename FieldT >
  Expr::IDSet 
  DensityFromSpeciesAndEnthalpy<FieldT>::
  register_local_expressions()
  {
    Expr::IDSet rootIDs;
    Expr::ExpressionFactory& factory = *(this->helper_.factory_);

    Expr::ExpressionID id;
    const TagNames& tagNames = TagNames::self();
    typedef typename Expr::PlaceHolder<FieldT>::Builder PlcHldr;

    factory.register_expression(new PlcHldr( this->densityOldTag_ ));
    factory.register_expression(new PlcHldr( pressureTag_ ));
    factory.register_expression(new PlcHldr( mmwOldTag_   ));
    factory.register_expression(new PlcHldr( hOldTag_     ));

    for(int i=0; i<this->nEq_; ++i)
    {
      factory.register_expression(new PlcHldr( this->rhoPhiTags_ [i] ));
      factory.register_expression(new PlcHldr( this->betaOldTags_[i] ));
    }

    factory.register_expression(new PlcHldr( yOldTags_[nSpec_-1] ));

    for(int i = 0; i<nSpec_; ++i)
    {
      factory.register_expression( new typename pokitt::SpeciesEnthalpy
                                        <FieldT>::Builder( hiTags_[i],
                                                            temperatureOldTag_,
                                                            i ));
    }

    // register an expression for the nth species, mixture molecular weight, enthalpy, and density. 
    id =             
    factory.register_expression( new typename pokitt::SpeciesN
                                      <FieldT>::Builder( yNewTags_[nSpec_-1],
                                                         yNewTags_,
                                                         pokitt::CLIPSPECN ));
    rootIDs.insert( id );

    id = 
    factory.register_expression( new typename
                                      pokitt::MixtureMolWeight
                                      <FieldT>::Builder( mmwNewTag_,
                                                         yNewTags_,
                                                         pokitt::MASS ));
    rootIDs.insert( id );

    id = 
    factory.register_expression( new typename
                                      pokitt::Enthalpy
                                      <FieldT>::Builder( hNewTag_,
                                                         temperatureNewTag_,
                                                         yNewTags_ ));
    rootIDs.insert( id );

    id = 
    factory.register_expression( new typename
                                      pokitt::Density
                                      <FieldT>::Builder( this->densityNewTag_,
                                                         temperatureNewTag_,
                                                         pressureTag_,
                                                         mmwNewTag_ ));
    rootIDs.insert( id );

    factory.register_expression( new typename
                                      pokitt::HeatCapacity_Cp
                                      <FieldT>::Builder( cpTag_,
                                                         temperatureOldTag_,
                                                         yOldTags_ ));

    // register expression for calculating the residual vector
    factory.register_expression( new typename Residual
                                      <FieldT>::Builder( this->residualTags_,
                                                         this->rhoPhiTags_,
                                                         this->phiOldTags_,
                                                         this->densityOldTag_ ));
    factory.register_expression( new typename DRhoDY
                                      <FieldT>::Builder( dRhodYTags_,
                                                         this->densityOldTag_,
                                                         mmwOldTag_ ));

    factory.register_expression( new typename DRhoDEnthalpy
                                      <FieldT>::Builder( dRhodHTag_,
                                                          this->densityOldTag_,
                                                          cpTag_,
                                                          temperatureOldTag_ ));

    factory.register_expression( new typename DRhoDTemperature
                                      <FieldT>::Builder( dRhodTempertureTag_,
                                                         this->densityOldTag_,
                                                         temperatureOldTag_ ));

    Expr::TagList dHdYiTags;
    const std::string enthName = "solver_h";
    for(int i=0; i<nSpec_-1; ++i){
      const std::string& specName = CanteraObjects::species_name(i);
      dHdYiTags.push_back( tagNames.derivative_tag(enthName, specName) );
    }

    factory.register_expression( new typename DEnthalpyDY
                                      <FieldT>::Builder( dHdYiTags,
                                                         hiTags_ ));

    // matrix assembly   ----------------------------
    // ----------------------------------------------
    using ScaledIdMatrix = Expr::matrix::ScaledIdentityMatrix<FieldT>;
    using SparseMatrix   = Expr::matrix::SparseMatrix        <FieldT>;

    using ScaledIdPtr = boost::shared_ptr<ScaledIdMatrix>;
    using SparsePtr   = boost::shared_ptr<SparseMatrix  >;
    using BasePtr     = boost::shared_ptr<Expr::matrix::AssemblerBase<FieldT>>;

    SparsePtr   phiVec     = boost::make_shared<SparseMatrix  >( "phi"                );
    SparsePtr   dRhodBeta  = boost::make_shared<SparseMatrix  >( "d(rho)/d(beta_j)"   );
    SparsePtr   dPhidBeta  = boost::make_shared<SparseMatrix  >( "d(phi_i)/d(beta_j)" );
    ScaledIdPtr rhoI       = boost::make_shared<ScaledIdMatrix>( "rho * [I]"          );

    // assemble phi vector (nEq_ x 1) and d(rho)/d(beta) (1 x nEq_)
    for(int i=0; i<this->nEq_; ++i){
      phiVec   ->template element<FieldT>(i,0) = this->phiOldTags_   [i];
      dRhodBeta->template element<FieldT>(0,i) = this->dRhodBetaTags_[i];
    }

    // assemble matrix corresponding to d(phi_i)/d(beta_j)
    // beta_i = phi_i for i < nEq_
    const int end = this->nEq_ - 1;
    for(int i=0; i<end; ++i){
      dPhidBeta->template element<double>(i  ,i) = 1;
      dPhidBeta->template element<FieldT>(end,i) = dHdYiTags[i];
    }

    // the last element is d(h)/d(T) = c_p
    dPhidBeta->template element<FieldT>(end,end) = cpTag_;

    rhoI->template scale<FieldT>() = this->densityOldTag_;

    phiVec   ->finalize();
    dRhodBeta->finalize();
    dPhidBeta->finalize();
    rhoI     ->finalize();

    // assemble residual jacobian
    BasePtr jacobian = rhoI * dPhidBeta + phiVec * dRhodBeta;

    // ----------------------------------------------

    factory.register_expression( new typename Expr::matrix::MatrixExpression
                                      <FieldT>::Builder( this->jacobianTags_,
                                                        jacobian ) );

    id =
    factory.register_expression( new typename NewtonUpdate
                                      <FieldT>::Builder( this->betaNewTags_,
                                                         this->residualTags_,
                                                         this->jacobianTags_,
                                                         this->betaOldTags_ ));
    rootIDs.insert( id );

    // clip updated species mass fractions
    typedef Expr::ClipValue<FieldT> Clipper;
    // const typename Clipper::Options clipOpt = Clipper::CLIP_MIN_ONLY;
    for(int i = 0; i<nSpec_-1; ++i)
    {
      const Expr::Tag clipTag = Expr::Tag(yNewTags_[i].name()+"_clip", Expr::STATE_NONE);
      factory.register_expression( new typename Clipper::Builder( clipTag, 0., 1. ) );
      factory.attach_modifier_expression( clipTag, yNewTags_[i] );
    }

    return rootIDs;
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensityFromSpeciesAndEnthalpy<FieldT>::
  update_other_fields(Expr::UintahFieldManager<FieldT>& fieldTManager)
  {
    // update old molecular weight
    FieldT& mmwOld = fieldTManager.field_ref(mmwOldTag_);
    mmwOld <<= fieldTManager.field_ref(mmwNewTag_);

    // update old nth species
    FieldT& ynOld = fieldTManager.field_ref(yOldTags_[nSpec_-1]);
    ynOld <<= fieldTManager.field_ref(yNewTags_[nSpec_-1]);
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void 
  DensityFromSpeciesAndEnthalpy<FieldT>::
  set_initial_guesses()
  {
    Expr::UintahFieldManager<FieldT>& fieldManager = this->helper_.fml_-> template field_manager<FieldT>();

    FieldT& density = fieldManager.field_ref( this->densityOldTag_ );
    density <<= rhoOld_->field_ref();

    FieldT& temperature = fieldManager.field_ref( temperatureOldTag_ );
    temperature <<= temperatureOld_->field_ref();

    FieldT& mmw = fieldManager.field_ref( mmwOldTag_ );
    mmw <<= mmwOld_->field_ref();

    FieldT& pressure = fieldManager.field_ref( pressureTag_ );
    pressure <<= pressure_->field_ref();

    for(int i=0; i<nSpec_-1; ++i){
      FieldT& yiGuess = fieldManager.field_ref( yOldTags_[i] );
      yiGuess <<= yOld_[i]->field_ref();

      FieldT& rhoYi = fieldManager.field_ref( rhoYTags_[i] );
      rhoYi <<= rhoY_[i]->field_ref();
    }
      FieldT& ynGuess = fieldManager.field_ref( yOldTags_[nSpec_-1] );
      ynGuess <<= yOld_[nSpec_-1]->field_ref();

      FieldT& hOld = fieldManager.field_ref( hOldTag_ );
      hOld <<= hOld_->field_ref();

      FieldT& rhoH = fieldManager.field_ref( rhoHTag_ );
      rhoH <<= rhoH_->field_ref();
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  void
  DensityFromSpeciesAndEnthalpy<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    typename Expr::Expression<FieldT>::ValVec& results = this->get_value_vec();


    this->newton_solve();

    Expr::FieldManagerList* fml = this->helper_.fml_;
    Expr::UintahFieldManager<FieldT>& fieldTManager = fml-> template field_manager<FieldT>();

    int j = 0;

    // copy local fields to fields visible to uintah
    // first two fields in vector of results are density and temperature
    FieldT& rho = *results[j];
    rho <<= fieldTManager.field_ref( this->densityNewTag_ );
    j++;

    FieldT& temperature = *results[j];
    temperature <<= fieldTManager.field_ref( this->temperatureNewTag_ );
    j++;

    // the next nSpec_-1 fields are d(rho)/d(y_i)
    for(int i=0; i<nSpec_-1; ++i){
      FieldT& dRhodYi = *results[j];
      dRhodYi <<=  fieldTManager.field_ref( this->dRhodYTags_[i] );
      j++;
    }
    // last field is d(rho)/d(h)
    FieldT& dRhodH = *results[j];
    dRhodH <<= fieldTManager.field_ref( this->dRhodHTag_ );
  }

  //--------------------------------------------------------------------

  template< typename FieldT >
  DensityFromSpeciesAndEnthalpy<FieldT>::
  Builder::Builder( const Expr::Tag&     rhoNewTag,
                    const Expr::Tag&     temperatureNewTag,
                    const Expr::TagList& dRhodYTags,
                    const Expr::Tag&     dRhodHTag,
                    const Expr::Tag&     rhoOldTag,
                    const Expr::TagList& rhoYTags,
                    const Expr::Tag&     rhoHTag,
                    const Expr::TagList& yOldTags,
                    const Expr::Tag&     hOldTag,
                    const Expr::Tag&     temperatureOldTag,
                    const Expr::Tag&     mmwOldTag,
                    const Expr::Tag&     pressureTag,
                    const double         rTol,
                    const unsigned       maxIter )
    : ExpressionBuilder( concatenate_vectors( tag_list( rhoNewTag , temperatureNewTag ),
                                              tag_list( dRhodYTags, dRhodHTag ) 
                                            )
                        ),
      rhoYTags_         ( rhoYTags          ),
      yOldTags_         ( yOldTags          ),
      rhoOldTag_        ( rhoOldTag         ),
      rhoHTag_          ( rhoHTag           ),
      hOldTag_          ( hOldTag           ),
      temperatureOldTag_( temperatureOldTag ),
      mmwOldTag_        ( mmwOldTag         ),
      pressureTag_      ( pressureTag       ),
      rtol_             ( rTol              ),
      maxIter_          ( maxIter           )
  {}

  //===================================================================


  // explicit template instantiation
  #include <spatialops/structured/FVStaggeredFieldTypes.h>
  template class DensityFromSpeciesAndEnthalpy<SpatialOps::SVolField>;

}
