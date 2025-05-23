/**
 *  \file   EnthalpyTransportEquation.cc
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2025 The University of Utah
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
 *
 */
#include <CCA/Components/Wasatch/Transport/EnthalpyTransportEquation.h>
#include <CCA/Components/Wasatch/Expressions/PrimVar.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionsOneSided.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSCoupling.h>
#ifdef HAVE_POKITT
#include <CCA/Components/Wasatch/Expressions/QuotientFunction.h>
#include <CCA/Components/Wasatch/Expressions/PressureMaterialDerivativePartial.h>
#include <CCA/Components/Wasatch/Expressions/PressureMaterialDerivative.h>
#include <CCA/Components/Wasatch/Expressions/AdiabaticIndex.h>
#include <pokitt/thermo/Temperature.h>
#include <pokitt/thermo/Enthalpy.h>
#include <pokitt/transport/ThermalCondMix.h>
#include <pokitt/thermo/HeatCapacity_Cp.h>
#include <pokitt/transport/HeatFlux.h>
#endif

namespace WasatchCore {

typedef SpatialOps::SVolField FieldT;  // this is the field type we will support for enthalpy transport

template< typename FaceT, typename GradT >
struct EnthalpyBoundaryTyper
{
  typedef SpatialOps::Divergence DivT;

  typedef typename SpatialOps::OperatorTypeBuilder<GradT, FieldT, FieldT>::type CellNeumannT;
  typedef typename SpatialOps::OperatorTypeBuilder<DivT, FaceT, FieldT>::type FaceNeumannT;

  typedef typename SpatialOps::NeboBoundaryConditionBuilder<CellNeumannT> CellNeumannBCOpT;
  typedef typename SpatialOps::NeboBoundaryConditionBuilder<FaceNeumannT> FaceNeumannBCOpT;

  typedef typename ConstantBCNew<FieldT,CellNeumannBCOpT>::Builder ConstantCellNeumannBC;
  typedef typename ConstantBCNew<FaceT,FaceNeumannBCOpT>::Builder ConstantFaceNeumannBC;
};

  class EnthDiffCoeff
   : public Expr::Expression<FieldT>
  {
    DECLARE_FIELDS(FieldT, thermCond_, cp_, turbVisc_)
    const double turbPr_;
    const bool isTurbulent_;

    EnthDiffCoeff( const Expr::Tag& thermCondTag,
                   const Expr::Tag& viscosityTag,
                   const Expr::Tag& turbViscTag,
                   const double turbPr )
    : Expr::Expression<FieldT>(),
      turbPr_      ( turbPr ),
      isTurbulent_ ( turbViscTag != Expr::Tag() )
    {
       this->set_gpu_runnable(true);
       thermCond_ = create_field_request<FieldT>(thermCondTag);
       cp_ = create_field_request<FieldT>(viscosityTag);
      if(isTurbulent_)  turbVisc_ = create_field_request<FieldT>(turbViscTag);
    }

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a EnthDiffCoeff expression
       *  @param resultTag the tag for the value that this expression computes
       *  @param thermCondTag the tag for the thermal conductivity
       *  @param cpTag the tag for the specific heat
       *  @param turbViscTag the tag for the turbulent viscosity.  If empty, no turbulent contribution is included.
       *  @param turbPrandtl the turbulent Prandtl number.
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& thermCondTag,
               const Expr::Tag& cpTag,
               const Expr::Tag& turbViscTag,
               const double turbPrandtl = 0.7 )
      : ExpressionBuilder( resultTag ),
        thermCondTag_( thermCondTag ),
        cpTag_       ( cpTag ),
        turbViscTag_ ( turbViscTag  ),
        turbPr_      ( turbPrandtl  )
      {}

      Expr::ExpressionBase* build() const{
        return new EnthDiffCoeff( thermCondTag_,cpTag_,turbViscTag_,turbPr_ );
      }

    private:
      const Expr::Tag thermCondTag_, cpTag_, turbViscTag_;
      const double turbPr_;
    };

    ~EnthDiffCoeff(){}

    void evaluate(){
      using namespace SpatialOps;
      FieldT& result = this->value();
      const FieldT& thermCond = thermCond_->field_ref();
      const FieldT& cp = cp_->field_ref();
      if( isTurbulent_ ) result <<= thermCond / cp + turbVisc_->field_ref()/turbPr_;
      else               result <<= thermCond / cp;
    }
  };

#ifdef HAVE_POKITT
  //---------------------------------------------------------------------------

  /* defines a TagList corresponding to species diffusive flux given a string
   * defining a direction and an Expr::Context
   */
  Expr::TagList species_diffusive_flux_tags( std::string dir,
                                             Expr::Context context )
  {
    assert( dir=="X" || dir=="Y" || dir=="Z" );

    Expr::TagList diffFluxTags;
    for( const std::string specName : CanteraObjects::species_names()){
      diffFluxTags.push_back(Expr::Tag(specName + TagNames::self().diffusiveflux + dir, context));
    }
    return diffFluxTags;
  }

  //---------------------------------------------------------------------------
#endif

  //===========================================================================

  EnthalpyTransportEquation::
  EnthalpyTransportEquation( const std::string enthalpyName,
                             Uintah::ProblemSpecP params,
                             Uintah::ProblemSpecP wasatchParams,
                             GraphCategories& gc,
                             const Expr::Tag densityTag,
                             const TurbulenceParameters& turbulenceParams,
                             std::set<std::string>& persistentFields )
  : ScalarTransportEquation<FieldT>( enthalpyName, params, gc, densityTag, turbulenceParams, persistentFields, false ),
    wasatchSpec_( wasatchParams ),
    diffCoeffTag_( enthalpyName+"_diffusivity", Expr::STATE_NONE ),
    temperatureTag_(TagNames::self().temperature),
    usePokitt_( params->findBlock("PropertiesFromPoKiTT") )
  {
    // todo: allow enthalpy to be parsed like TotalInternalEnergy. That way, most of the mess below can be removed.
    const TagNames& tags = TagNames::self();
    Expr::Tag lambdaTag, cpTag;
    Expr::ExpressionFactory& solnFactory = *gc[ADVANCE_SOLUTION]->exprFactory;
    Expr::ExpressionFactory& icFactory   = *gc[INITIALIZATION  ]->exprFactory;

    Uintah::ProblemSpecP lambdaParams =  params->findBlock("ThermalConductivity");
    Uintah::ProblemSpecP cpParams     =  params->findBlock("HeatCapacity"       );


    const bool propertiesSpecified = ( lambdaParams != nullptr && cpParams != nullptr);

#   ifdef HAVE_POKITT
    specParams_ = wasatchSpec_->findBlock("SpeciesTransportEquations");

    // determine if species diffusion
    if( usePokitt_ ){
      if( propertiesSpecified ){
        std::ostringstream msg;
        msg << "Specification of a 'TransportEquation equation=\"enthalpy\"' block requires" << std::endl
            << "sub-blocks for 'ThermalConductivity' and 'HeatCapacity' or a"                << std::endl
            << "'PropertiesFromPoKiTT' sub-block, but not both. Please remove either the "   << std::endl
            << "'PropertiesFromPoKiTT' sub-block, or the  'ThermalConductivity' and"         << std::endl
            << "'HeatCapacity' sub-blocks."
            << std::endl;
        throw Uintah::ProblemSetupException(msg.str(), __FILE__, __LINE__ );
      }
      lambdaTag = tags.thermalConductivity;
      cpTag     = tags.heatCapacity;

      Expr::Context contextN = flowTreatment_ == LOWMACH ? Expr::STATE_N : Expr::STATE_NONE;
      yiTags_.clear(); yiNP1Tags_.clear(); yiInitTags_.clear();
      for( int i=0; i<CanteraObjects::number_species(); ++i ){
        const std::string specName = CanteraObjects::species_name(i);
        yiTags_    .push_back( Expr::Tag( specName, contextN         ) );
        yiNP1Tags_ .push_back( Expr::Tag( specName, Expr::STATE_NP1  ) );
        yiInitTags_.push_back( Expr::Tag( specName, Expr::STATE_NONE ) );
      }
      const Expr::TagList& yiTags = flowTreatment_ == LOWMACH ? yiNP1Tags_ : yiTags_;
      // temperature calculation
      const Expr::Tag temperatureTag = tags.temperature;

      typedef pokitt::HeatCapacity_Cp<FieldT>::Builder Cp;
      typedef pokitt::ThermalConductivity<FieldT>::Builder ThermCond;

      solnFactory.register_expression( scinew Cp( cpTag, temperatureTag, yiTags ) );
      solnFactory.register_expression( scinew ThermCond( lambdaTag, temperatureTag,  yiTags, tags.mixMW ) );

      if( flowTreatment_ == LOWMACH ){
        icFactory.register_expression( scinew Cp( cpTag, temperatureTag, yiInitTags_ ) );
        icFactory.register_expression( scinew ThermCond( lambdaTag, temperatureTag,  yiInitTags_, tags.mixMW ) );
      }
    }
      else
#   endif
    if( propertiesSpecified ){
      lambdaTag = parse_nametag( lambdaParams->findBlock("NameTag") );
      cpTag     = parse_nametag( cpParams    ->findBlock("NameTag") );
      // if not calculated by PoKiTT, ensure that heat capcity and thermal conductivity are
      // specified in the input file
      }
      else{
        std::ostringstream msg;
        msg << "Specification of a 'TransportEquation equation=\"enthalpy\"' block requires" << std::endl
            << "sub-blocks for 'ThermalConductivity' and 'HeatCapacity' or a"                << std::endl
            << "'PropertiesFromPoKiTT' sub-block"
            << std::endl;
        throw Uintah::ProblemSetupException(msg.str(), __FILE__, __LINE__ );
      }

    const Expr::Tag turbViscTag = enableTurbulence_ ? tags.turbulentviscosity : Expr::Tag();
    typedef EnthDiffCoeff::Builder DiffCoeff;
    solnFactory.register_expression( scinew DiffCoeff( diffCoeffTag_, lambdaTag, cpTag, turbViscTag, turbulenceParams.turbPrandtl ) );
    if(flowTreatment_ == LOWMACH ){
      icFactory.register_expression( scinew DiffCoeff( diffCoeffTag_, lambdaTag, cpTag, turbViscTag, turbulenceParams.turbPrandtl ) );
    }

    setup();
  }

  //---------------------------------------------------------------------------

  EnthalpyTransportEquation::~EnthalpyTransportEquation()
  {}

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  setup_diffusive_flux( FieldTagInfo& info )
  {
    if( flowTreatment_ == LOWMACH ){
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      if(enableTurbulence_){
        register_diffusive_flux_expressions( ADVANCE_SOLUTION, info    , primVarTag_, densityTag_, Expr::STATE_NONE, ""               );
        register_diffusive_flux_expressions( ADVANCE_SOLUTION, infoNP1_, primVarTag_, densityTag_, Expr::STATE_NONE,  "-NP1-estimate" );
      }

      else{
        Expr::ExpressionFactory& icFactory = *gc_[INITIALIZATION]->exprFactory;
        register_diffusive_flux_expressions( INITIALIZATION  , infoInit_, primVarInitTag_, densityInitTag_, Expr::STATE_NONE, "" );
        register_diffusive_flux_expressions( ADVANCE_SOLUTION, infoNP1_ , primVarNP1Tag_ , densityNP1Tag_ , Expr::STATE_NP1 , "" );

        const std::set<FieldSelector> fsSet = {DIFFUSIVE_FLUX_X, DIFFUSIVE_FLUX_Y, DIFFUSIVE_FLUX_Z};
        for( FieldSelector fs : fsSet ){
         if( infoNP1_.find(fs) != infoNP1_.end() ){
           const std::string diffFluxName = infoNP1_[fs].name();
           info[fs] = Expr::Tag(diffFluxName, Expr::STATE_N);
           persistentFields_.insert( diffFluxName );

           // Force diffusive flux expression on initialization graph.
           const Expr::ExpressionID id = icFactory.get_id( infoInit_[fs] );
           gc_[INITIALIZATION]->rootIDs.insert(id);
          }
        }// for( FieldSelector fs : fsSet )

        // Register placeholders for diffusive flux parameters at STATE_N
        register_diffusive_flux_placeholders<FieldT>( factory, info );
      }// else -- (enableTurbulence_ == false)
    }// if( flowTreatment_ == LOWMACH )

    else{
      register_diffusive_flux_expressions( ADVANCE_SOLUTION, info, primVarTag_, densityTag_, Expr::STATE_NONE, "" );
    }
  }

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  register_diffusive_flux_expressions( const Category      cat,
                                       FieldTagInfo&       info,
                                       Expr::Tag           primVarTag,
                                       Expr::Tag           densityTag,
                                       const Expr::Context context,
                                       const std::string   suffix )
  {
    Expr::ExpressionFactory& factory = *gc_[cat]->exprFactory;
    const TagNames& tagNames = TagNames::self();
    const std::string primVarName = this->primVarTag_.name();

    const Expr::Tag xDiffFluxTag( primVarName + tagNames.diffusiveflux + "X" + suffix, context );
    const Expr::Tag yDiffFluxTag( primVarName + tagNames.diffusiveflux + "Y" + suffix, context );
    const Expr::Tag zDiffFluxTag( primVarName + tagNames.diffusiveflux + "Z" + suffix, context );

    bool doX = false, doY = false, doZ = false;

#   ifdef HAVE_POKITT

    if( specParams_ ){
      if( params_->findBlock("DiffusiveFlux") ){
        std::ostringstream msg;
        msg << "ERROR: When using species transport, the enthalpy diffusive flux will be automatically calculated.\n"
            << "       Do not specify it in your input file." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }

      if( enableTurbulence_ ){
        throw Uintah::ProblemSetupException( "Turbulent transport of enthalpy + species transport not yet supported.",
                                             __FILE__, __LINE__ );
      }

      typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<FieldT>::XFace >::Builder XFlux;
      typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<FieldT>::YFace >::Builder YFlux;
      typedef pokitt::HeatFlux< typename SpatialOps::FaceTypes<FieldT>::ZFace >::Builder ZFlux;

      Expr::TagList specEnthTags, specEngyTags;
      const Expr::Tag thermCondTag = tagNames.thermalConductivity;
      const Expr::TagList emptyTagList;

      typedef pokitt::SpeciesEnthalpy<FieldT>::Builder SpecEnth;
      for( int i=0; i<CanteraObjects::number_species(); ++i ){
        const std::string specName = CanteraObjects::species_name(i);
        specEnthTags.push_back( Expr::Tag( specName+"_"+tagNames.enthalpy.name(), Expr::STATE_NONE ) );

        factory.register_expression( scinew SpecEnth( specEnthTags[i], temperatureTag_, i ) );
      }

      const Expr::TagList xSpecFluxTags = species_diffusive_flux_tags("X",context);
      const Expr::TagList ySpecFluxTags = species_diffusive_flux_tags("Y",context);
      const Expr::TagList zSpecFluxTags = species_diffusive_flux_tags("Z",context);

      for( Uintah::ProblemSpecP diffFluxParams=specParams_->findBlock("DiffusiveFlux");
          diffFluxParams != nullptr;
          diffFluxParams=diffFluxParams->findNextBlock("DiffusiveFlux") )
      {
        /* If a "MixtureAveraged" sub-block is found in the species diffusion block,
         * then we may have differential diffusion and we must account for its effects
         * on the heat flux. If a "MixtureAveraged" sub-block is not found, an empty
         * TagList (emptyTagList) is used in place of species diffusive fluxes and pure
         * species enthalpies.
         */
        Uintah::ProblemSpecP mixAvgParams = diffFluxParams->findBlock("MixtureAveraged");

        std::string direction;
        diffFluxParams->getAttribute("direction",direction);
        for( std::string::iterator it = direction.begin(); it != direction.end(); ++it ){
          const std::string dir(1,*it);
          if( dir == "X" ){
            doX = true;
            const Expr::ExpressionID id =
            factory.register_expression( scinew XFlux( xDiffFluxTag,
                                                       temperatureTag_,
                                                       thermCondTag,
                                                       mixAvgParams ? specEnthTags  : emptyTagList,
                                                       mixAvgParams ? xSpecFluxTags : emptyTagList ) );

            if(cat == ADVANCE_SOLUTION) factory.cleave_from_children( id );

            info[DIFFUSIVE_FLUX_X] = xDiffFluxTag;
          }
          else if( dir == "Y" ){
            doY = true;
            const Expr::ExpressionID id =
            factory.register_expression( scinew YFlux( yDiffFluxTag,
                                                       temperatureTag_,
                                                       thermCondTag,
                                                       mixAvgParams ? specEnthTags  : emptyTagList,
                                                       mixAvgParams ? ySpecFluxTags : emptyTagList ) );

            if(cat == ADVANCE_SOLUTION) factory.cleave_from_children( id );

            info[DIFFUSIVE_FLUX_Y] = yDiffFluxTag;
          }
          else if( dir == "Z" ){
            doZ = true;
            const Expr::ExpressionID id =
            factory.register_expression( scinew ZFlux( zDiffFluxTag,
                                                       temperatureTag_,
                                                       thermCondTag,
                                                       mixAvgParams ? specEnthTags  : emptyTagList,
                                                       mixAvgParams ? zSpecFluxTags : emptyTagList ) );

            if(cat == ADVANCE_SOLUTION) factory.cleave_from_children( id );

            info[DIFFUSIVE_FLUX_Z] = zDiffFluxTag;
          }
        }
      }

      // todo: this will need to be changed when turbulence is enabled
      if(context == Expr::STATE_NP1){
        if(doX) speciesInfoNP1_[DIFFUSIVE_FLUX_X] = xSpecFluxTags;
        if(doY) speciesInfoNP1_[DIFFUSIVE_FLUX_Y] = ySpecFluxTags;
        if(doZ) speciesInfoNP1_[DIFFUSIVE_FLUX_Z] = zSpecFluxTags;
      }
    }
    else
#   endif
    {
      // jcs how will we determine if we have each direction on???
      doX=true, doY=true, doZ=true;

      typedef DiffusiveFlux< typename SpatialOps::FaceTypes<FieldT>::XFace >::Builder XFlux;
      typedef DiffusiveFlux< typename SpatialOps::FaceTypes<FieldT>::YFace >::Builder YFlux;
      typedef DiffusiveFlux< typename SpatialOps::FaceTypes<FieldT>::ZFace >::Builder ZFlux;

      if( doX ){
        factory.register_expression( new XFlux( xDiffFluxTag,
                                                primVarTag,
                                                diffCoeffTag_,
                                                turbDiffTag_,
                                                densityTag ) );
        info[DIFFUSIVE_FLUX_X] = xDiffFluxTag;
      }
      if( doY ){
        factory.register_expression( new YFlux( yDiffFluxTag,
                                                primVarTag,
                                                diffCoeffTag_,
                                                turbDiffTag_,
                                                densityTag ) );
        info[DIFFUSIVE_FLUX_Y] = yDiffFluxTag;
      }
      if( doZ ){
        factory.register_expression( new ZFlux( zDiffFluxTag,
                                                primVarTag,
                                                diffCoeffTag_,
                                                turbDiffTag_,
                                                densityTag ) );
        info[DIFFUSIVE_FLUX_Z] = zDiffFluxTag;
      }
    }
  }

  //---------------------------------------------------------------------------
  
  void
  EnthalpyTransportEquation::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);

    // set up the extra fields for setting BCs on primitives
    const Expr::Tag temporaryHTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryRhoTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryRhoHTag( "temporary_rho" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );

    if( !( advSlnFactory.have_entry( temporaryHTag ) ) ){
      advSlnFactory.register_expression( new Expr::ConstantExpr<FieldT>::Builder( temporaryHTag, 0.0 ) );
    }
    if( !( initFactory.have_entry( temporaryHTag ) ) ){
      initFactory.register_expression( new Expr::ConstantExpr<FieldT>::Builder( temporaryHTag, 0.0 ) );
    }
    if( !( advSlnFactory.have_entry( temporaryRhoHTag ) ) ){
      typedef ExprAlgebra<FieldT>::Builder RhoH;
      advSlnFactory.register_expression( new RhoH( temporaryRhoHTag, Expr::tag_list( temporaryRhoTag, temporaryHTag ), ExprAlgebra<FieldT>::PRODUCT ) );
    }
    if( !( initFactory.have_entry( temporaryRhoHTag ) ) ){
      typedef ExprAlgebra<FieldT>::Builder RhoH;
      initFactory.register_expression( new RhoH( temporaryRhoHTag, Expr::tag_list( temporaryRhoTag, temporaryHTag ), ExprAlgebra<FieldT>::PRODUCT ) );
    }

    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() ){
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;

      // a lambda to make decorated tags for boundary condition expressions for a this boundary spec
      //
      // param: exprName: a string, the name of the field on which we will impose the boundary condition
      // param: description: a string describing the boundary condition, such as "neumann-zero-for-outflow" or "dirichlet-for-inflow"
      // param: direction: a string for the direction of the boundary face, such as "X", "Y", or "Z"
      auto get_decorated_tag = [&myBndSpec]( const std::string exprName, const std::string description, const std::string direction )
      {
        return Expr::Tag( exprName + "_STATE_NONE_" + description + "_bc_" + myBndSpec.name + "_" + direction + "dir", Expr::STATE_NONE );
      };

      // for variable density problems, we must ALWAYS guarantee proper boundary conditions for
      // (rho h)_{n+1}. Since we apply bcs on (rho h) at the bottom of the graph, we can't apply
      // the same bcs on (rho h) (time advanced). Hence, we set the rhs to zero always :)
      if( !myBndSpec.has_field(rhs_name()) ){
        const BndCondSpec rhsBCSpec = { rhs_name(), "none", 0.0, DIRICHLET, DOUBLE_TYPE };
        bcHelper.add_boundary_condition(bndName, rhsBCSpec);
      }

      switch ( myBndSpec.type ){
        case WALL:
        {
          const Uintah::Patch::FaceType& face = myBndSpec.face;
          const std::string dir = (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus) ? "X"
                                : (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus) ? "Y"
                                : (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus) ? "Z" : "INVALID";

          const std::string diffFluxName( primVarTag_.name() + "_diffFlux_" + dir );
          const std::string convFluxName( this->solnVarName_ + "_convFlux_" + dir );

          // set zero convective and diffusive fluxes
          const BndCondSpec diffFluxBCSpec = { diffFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
          const BndCondSpec convFluxBCSpec = { convFluxName, "none" , 0.0, DIRICHLET, DOUBLE_TYPE };
          bcHelper.add_boundary_condition( bndName, diffFluxBCSpec );
          bcHelper.add_boundary_condition( bndName, convFluxBCSpec );
        }
        break;


        case OUTFLOW:
        case OPEN:{

          const std::string normalConvFluxNameBase = this->solnVarName_ + "_convFlux_"; // normal convective flux
          const std::string normalDiffFluxNameBase = primVarTag_.name() + "_diffFlux_"; // normal diffusive flux

          Expr::Tag neumannZeroConvectiveFluxTag,
                    neumannZeroDiffFluxTag,
                    neumannZeroRhoHTag;
          std::string normalConvFluxName,
                      normalDiffFluxName;

          // build boundary conditions for x, y, and z faces
          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              std::string dir = "X";
              typedef EnthalpyBoundaryTyper<XFaceT, SpatialOps::GradientX> BCTypes;

              normalConvFluxName = normalConvFluxNameBase + dir;
              normalDiffFluxName = normalDiffFluxNameBase + dir;

              neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "neumann-zero", dir );
              neumannZeroRhoHTag           = get_decorated_tag( this->solnVarName_, "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroRhoHTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroRhoHTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
            }
            break;
            case Uintah::Patch::yminus:
            case Uintah::Patch::yplus:
            {
              std::string dir = "Y";
              typedef EnthalpyBoundaryTyper<YFaceT, SpatialOps::GradientY> BCTypes;

              normalConvFluxName = normalConvFluxNameBase + dir;
              normalDiffFluxName = normalDiffFluxNameBase + dir;

              neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "neumann-zero", dir );
              neumannZeroRhoHTag           = get_decorated_tag( this->solnVarName_, "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroRhoHTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroRhoHTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
            }
            break;
            case Uintah::Patch::zminus:
            case Uintah::Patch::zplus:
            {
              std::string dir = "Z";
              typedef EnthalpyBoundaryTyper<ZFaceT, SpatialOps::GradientZ> BCTypes;

              normalConvFluxName = normalConvFluxNameBase + dir;
              normalDiffFluxName = normalDiffFluxNameBase + dir;

              neumannZeroDiffFluxTag       = get_decorated_tag( normalDiffFluxName, "neumann-zero", dir );
              neumannZeroConvectiveFluxTag = get_decorated_tag( normalConvFluxName, "neumann-zero", dir );
              neumannZeroRhoHTag           = get_decorated_tag( this->solnVarName_, "neumann-zero", dir );

              if( !advSlnFactory.have_entry( neumannZeroDiffFluxTag       ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroDiffFluxTag      , 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroRhoHTag           ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroRhoHTag           ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroRhoHTag          , 0.0 ) );
            }
            break;
            default:
              break;
          }

          // make boundary condition specifications, connecting the name of the field (first) and the name of the modifier expression (second)
          BndCondSpec diffFluxSpec = {normalDiffFluxName, neumannZeroDiffFluxTag      .name(), 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec convFluxSpec = {normalConvFluxName, neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec rhoYSpec     = {this->solnVarName_, neumannZeroRhoHTag          .name(), 0.0, NEUMANN, FUNCTOR_TYPE};

          // add boundary condition specifications to this boundary
          bcHelper.add_boundary_condition( bndName, diffFluxSpec );
          bcHelper.add_boundary_condition( bndName, convFluxSpec );
          bcHelper.add_boundary_condition( bndName, rhoYSpec     );
        }
        break;

        case VELOCITY:{

          Expr::Tag bcCopiedRhoHTag;

          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              std::string dir = "X";
              typedef typename BCCopier<FieldT>::Builder CopiedCellBC;
              bcCopiedRhoHTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedRhoHTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
              if( !advSlnFactory.have_entry( bcCopiedRhoHTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
            }
            break;
            case Uintah::Patch::yminus:
            case Uintah::Patch::yplus:
            {
              std::string dir = "Y";
              typedef typename BCCopier<FieldT>::Builder CopiedCellBC;
              bcCopiedRhoHTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedRhoHTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
              if( !advSlnFactory.have_entry( bcCopiedRhoHTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
            }
            break;
            case Uintah::Patch::zminus:
            case Uintah::Patch::zplus:
            {
              std::string dir = "Z";
              typedef typename BCCopier<FieldT>::Builder CopiedCellBC;
              bcCopiedRhoHTag = get_decorated_tag( this->solnVarTag_.name(), "bccopy-for-inflow", dir );
              if( !initFactory  .have_entry( bcCopiedRhoHTag ) ) initFactory  .register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
              if( !advSlnFactory.have_entry( bcCopiedRhoHTag ) ) advSlnFactory.register_expression( new CopiedCellBC( bcCopiedRhoHTag, temporaryRhoHTag ) );
            }
            break;
            default:
              break;
          }

          BndCondSpec rhoYDirichletBC = {this->solnVarTag_.name(), bcCopiedRhoHTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
          bcHelper.add_boundary_condition( bndName, rhoYDirichletBC );
        }
        break;

        case USER:
        default:
          // do nothing
          break;
      }
    }
  }

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;

    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;

    // multiply the initial condition by the volume fraction for embedded geometries
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( vNames.has_embedded_geometry() ) {

      //create modifier expression
      typedef ExprAlgebra<FieldT> ExprAlgbr;
      const Expr::TagList theTagList = tag_list( vNames.vol_frac_tag<SVolField>() );
      const Expr::Tag modifierTag = Expr::Tag( this->solution_variable_name() + "_init_cond_modifier", Expr::STATE_NONE);
      factory.register_expression( new typename ExprAlgbr::Builder( modifierTag,
                                                                    theTagList,
                                                                    ExprAlgbr::PRODUCT,
                                                                    true ) );

      factory.attach_modifier_expression( modifierTag, initial_condition_tag() );
    }

    if( factory.have_entry(initial_condition_tag()) ){
      bcHelper.apply_boundary_condition<FieldT>( initial_condition_tag(), taskCat );
    }
    if( factory.have_entry(primVarTag_) ){
      bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat, true );
    }
    // bcs for hard inflow - set primitive and conserved variables
    const Expr::Tag temporaryHTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
    bcHelper.apply_boundary_condition<FieldT>( temporaryHTag, taskCat, true );
  }

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  setup_source_terms( FieldTagInfo& info, Expr::TagList& srcTags )
  {
    // see if we have radiation - if so, plug it in
    if( params_->findBlock("RadiativeSourceTerm") ){
      info[SOURCE_TERM] = parse_nametag( params_->findBlock("RadiativeSourceTerm")->findBlock("NameTag") );
    }

    // populate any additional source terms pushed on this equation
    ScalarTransportEquation<FieldT>::setup_source_terms( info, srcTags );
  }

  //---------------------------------------------------------------------------

  void
  EnthalpyTransportEquation::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const bool isLowMach = flowTreatment_ == LOWMACH;
    const bool setOnExtraOnly = isLowMach;

    const Category taskCat = ADVANCE_SOLUTION;
    bcHelper.apply_boundary_condition<FieldT>( solution_variable_tag(), taskCat );
    bcHelper.apply_boundary_condition<FieldT>( rhs_tag(), taskCat, true ); // apply the rhs bc directly inside the extra cell

    // bcs for hard inflow - set primitive and conserved variables
    bcHelper.apply_boundary_condition<FieldT>( primVarTag_, taskCat );
    const Expr::Tag temporaryYTag( "temporary_" + this->primVarTag_.name() + "_for_bcs", Expr::STATE_NONE );
    bcHelper.apply_boundary_condition<FieldT>( temporaryYTag, taskCat, true );

    const std::string normalConvFluxName_nodir = this->solnVarName_ + "_convFlux_";
    bcHelper.apply_boundary_condition<XFaceT>(Expr::Tag(normalConvFluxName_nodir + 'X', Expr::STATE_NONE), taskCat, setOnExtraOnly);
    bcHelper.apply_boundary_condition<YFaceT>(Expr::Tag(normalConvFluxName_nodir + 'Y', Expr::STATE_NONE), taskCat, setOnExtraOnly);
    bcHelper.apply_boundary_condition<ZFaceT>(Expr::Tag(normalConvFluxName_nodir + 'Z', Expr::STATE_NONE), taskCat, setOnExtraOnly);



    // if the flow treatment is low-Mach, diffusive fluxes for STATE_NP1 are used to estimate div(u) so we must
    // set boundary conditions at STATE_NP1 as well as STATE_NONE when initial conditions are set
    const Expr::Context diffFluxContext = isLowMach ? Expr::STATE_NP1 : Expr::STATE_NONE;
    const std::string normalDiffFluxName_nodir = this->solnVarTag_.name() + "_diffFlux_";
    bcHelper.apply_boundary_condition<XFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'X', diffFluxContext ), taskCat, setOnExtraOnly);
    bcHelper.apply_boundary_condition<YFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'Y', diffFluxContext ), taskCat, setOnExtraOnly);
    bcHelper.apply_boundary_condition<ZFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'Z', diffFluxContext ), taskCat, setOnExtraOnly);

    if(isLowMach){
      const Category initCat = INITIALIZATION;
      const Expr::Context initContext = Expr::STATE_NONE;
      bcHelper.apply_boundary_condition<XFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'X', initContext), initCat, setOnExtraOnly);
      bcHelper.apply_boundary_condition<YFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'Y', initContext), initCat, setOnExtraOnly);
      bcHelper.apply_boundary_condition<ZFaceT>(Expr::Tag(normalDiffFluxName_nodir + 'Z', initContext), initCat, setOnExtraOnly);
    }
  }

  //------------------------------------------------------------------

  Expr::ExpressionID
  EnthalpyTransportEquation::
  setup_rhs( FieldTagInfo& info,
             const Expr::TagList& srcTags )
  {
    typedef typename ScalarRHS<FieldT>::Builder RHSBuilder;
    typedef typename ScalarEOSCoupling<FieldT>::Builder ScalarEOSBuilder;
    typedef typename PrimVar<FieldT,SVolField>::Builder PrimVar;
    Expr::ExpressionFactory& solnFactory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    Expr::ExpressionFactory& initFactory = *gc_[INITIALIZATION  ]->exprFactory;
    const TagNames& tagNames = TagNames::self();

    info[PRIMITIVE_VARIABLE] = primVarTag_;

    if( flowTreatment_ == COMPRESSIBLE ){
      solnFactory.register_expression( scinew PrimVar( primVarTag_, solnVarTag_, densityTag_ ) );
    }

    Expr::ExpressionID rhsID =
    solnFactory.register_expression( scinew RHSBuilder( rhsTag_, info, srcTags, densityTag_, isConstDensity_, isStrong_, tagNames.divrhou ) );

    // for variable density flows:
    if( flowTreatment_ == LOWMACH ){
      infoNP1_ [PRIMITIVE_VARIABLE]  = primVarNP1Tag_;
      infoInit_[PRIMITIVE_VARIABLE]  = primVarInitTag_;

      EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
      if( vNames.has_embedded_geometry() ){
        infoNP1_ [VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
        infoNP1_ [AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
        infoNP1_ [AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
        infoNP1_ [AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();

        infoInit_[VOLUME_FRAC] = vNames.vol_frac_tag<SVolField>();
        infoInit_[AREA_FRAC_X] = vNames.vol_frac_tag<XVolField>();
        infoInit_[AREA_FRAC_Y] = vNames.vol_frac_tag<YVolField>();
        infoInit_[AREA_FRAC_Z] = vNames.vol_frac_tag<ZVolField>();
      }

      typedef typename Expr::PlaceHolder<FieldT>::Builder PlaceHolder;
      if(isStrong_){
        solnFactory.register_expression( scinew PrimVar( primVarNP1Tag_, this->solnvar_np1_tag(), densityNP1Tag_ ) );
        solnFactory.register_expression( scinew PlaceHolder(primVarTag_) );
      }

      const Expr::Tag scalEOSTag = Expr::Tag(primVarTag_.name() + "_EOS_Coupling", Expr::STATE_NONE);
      const Expr::Tag dRhoDfTag  = tagNames.derivative_tag( densityTag_, primVarTag_ );

      // register an expression for divu. divu is just a constant expression to which we add the
      // necessary couplings from the scalars that represent the equation of state.
      typedef typename Expr::ConstantExpr<SVolField>::Builder ConstBuilder;

      if( !solnFactory.have_entry( tagNames.divu ) ) { // if divu has not been registered yet, then register it!
        solnFactory.register_expression( scinew ConstBuilder(tagNames.divu, 0.0)); // set the value to zero so that we can later add sources to it
      }
      if( !initFactory.have_entry( tagNames.divu ) ) {
        initFactory.register_expression( scinew ConstBuilder(tagNames.divu, 0.0));
      }

      // For enthalpy transport, we treat derivative, DP/Dt as a source term. Because DP/Dt depends on velocity divergence (divu),
      // which is calculated from non-convective scalar RHSs, we need to split DP/Dt into 2 parts:
      //    - a divergence-free portion (with associated tag 'partialDPDtTag')
      //    - the part that depends on divu, which is not explicitly calculated
      //
      // 'partialDPDtTag' is included in scalarEOSsrcTags, but is not added the RHS of the enthalpy transport equation. Instead,
      // DP/Dt is calculated from the divergence-free part of DP/Dt at STATE_NP1, and added to the enthalpy RHS during the
      // subsequent time step, so a placeholder for DP/Dt is registered for STATE_N. For now, we assume DP/Dt is zero unless we
      // are transporting species (rather than mixture fraction).

      Expr::TagList scalarEOSsrcTags = srcTags;

#ifdef HAVE_POKITT
    if(this->specParams_ != nullptr){
      typedef AdiabaticIndex                   <FieldT>::Builder AdiabaticIndex;
      typedef PressureMaterialDerivativePartial<FieldT>::Builder DPDtPartial;
      typedef PressureMaterialDerivative       <FieldT>::Builder DPDt;

      const Expr::Tag gammaTag           = tagNames.adiabaticIndex;
      const Expr::Tag fullDPDtTag        = tagNames.DPDt;
      const Expr::Tag fullDPDtNP1Tag     = tagNames.DPDtNP1;
      const Expr::Tag partialDPDtTag     = tagNames.partialDPDt;
      const Expr::Tag divuTag            = tagNames.divu;
      const Expr::Tag divuModifierTag    = Expr::Tag( divuTag.name()+"_enthalpy_modifier", Expr::STATE_NONE );
      const Expr::Tag heatCapacityTag    = tagNames.heatCapacity;
      const Expr::Tag mixMWTag           = tagNames.mixMW;

      // tags only needed at initialization
      const Expr::Tag fullDPDtInitTag = Expr::Tag( fullDPDtTag.name(), Expr::STATE_NONE );

      FieldTagListInfo speciesInfoInit;

      speciesInfoNP1_[PRIMITIVE_VARIABLE] = yiNP1Tags_ ;
      speciesInfoInit[PRIMITIVE_VARIABLE] = yiInitTags_;

      // set species diffusive flux info corresponding to the initial condition
      const std::set<FieldSelector> fsSet = {DIFFUSIVE_FLUX_X, DIFFUSIVE_FLUX_Y, DIFFUSIVE_FLUX_Z};
      for( FieldSelector fs : fsSet ){

       if( info.find(fs) != info.end() ){
         speciesInfoInit[fs] = Expr::TagList();
         for(const Expr::Tag& specDiffFluxTag : speciesInfoNP1_[fs] )
           speciesInfoInit[fs].push_back( Expr::Tag(specDiffFluxTag.name(), Expr::STATE_NONE) );
        }
      }// for( FieldSelector fs : fsSet )

        const bool kineticsEnabled = specParams_->findBlock("DetailedKinetics");

        Expr::TagList specEnthTags;

        for( int i=0; i<CanteraObjects::number_species(); ++i ){
          const std::string specName = CanteraObjects::species_name(i);
          specEnthTags.push_back( Expr::Tag( specName+"_"+tagNames.enthalpy.name(), Expr::STATE_NONE ) );

          if(kineticsEnabled){
            const std::string rrName = "rr_" + CanteraObjects::species_name(i);

            speciesInfoNP1_[SOURCE_TERM].push_back(Expr::Tag( rrName, Expr::STATE_NP1  ));
            speciesInfoInit[SOURCE_TERM].push_back(Expr::Tag( rrName, Expr::STATE_NONE ));
          }
        }

      solnFactory.register_expression( scinew AdiabaticIndex( gammaTag, heatCapacityTag, mixMWTag) );
      initFactory.register_expression( scinew AdiabaticIndex( gammaTag, heatCapacityTag, mixMWTag) );

      solnFactory.register_expression( scinew DPDtPartial( partialDPDtTag,
                                                           densityNP1Tag_,
                                                           temperatureTag_,
                                                           gammaTag,
                                                           specEnthTags,
                                                           infoNP1_,
                                                           speciesInfoNP1_ ) );

      initFactory.register_expression( scinew DPDtPartial( partialDPDtTag,
                                                           densityInitTag_,
                                                           temperatureTag_,
                                                           gammaTag,
                                                           specEnthTags,
                                                           infoInit_,
                                                           speciesInfoInit ) );

      const Expr::Tag thermoPNP1Tag  = tagNames.thermodynamicPressureNP1;
      const Expr::Tag thermoPInitTag = Expr::Tag(thermoPNP1Tag.name(), Expr::STATE_NONE);

      Expr::ExpressionID fullDPDtID;

      fullDPDtID =
      solnFactory.register_expression( scinew DPDt( fullDPDtNP1Tag,
                                                    partialDPDtTag,
                                                    gammaTag,
                                                    thermoPNP1Tag,
                                                    divuTag ));

      gc_[ADVANCE_SOLUTION]->rootIDs.insert(fullDPDtID);

      fullDPDtID =
      initFactory.register_expression( scinew DPDt( fullDPDtInitTag,
                                                    partialDPDtTag,
                                                    gammaTag,
                                                    thermoPInitTag,
                                                    divuTag ));

      gc_[INITIALIZATION]->rootIDs.insert(fullDPDtID);

      solnFactory.register_expression( scinew PlaceHolder(fullDPDtTag) );

      solnFactory.attach_dependency_to_expression(fullDPDtTag, rhsTag_);

      persistentFields_.insert( fullDPDtTag.name() );
    }
#endif

      solnFactory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoNP1_ , scalarEOSsrcTags, densityNP1Tag_ , dRhoDfTag, isStrong_) );
      initFactory.register_expression( scinew ScalarEOSBuilder( scalEOSTag, infoInit_, scalarEOSsrcTags, densityInitTag_, dRhoDfTag, isStrong_) );

      solnFactory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
      initFactory.attach_dependency_to_expression(scalEOSTag, tagNames.divu);
  }

    return rhsID;
  }

  //------------------------------------------------------------------

  Expr::ExpressionID
  EnthalpyTransportEquation::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    // initial condition for enthalpy
#ifdef HAVE_POKITT
   if( usePokitt_ ){
      const TagNames& tags = TagNames::self();

      typedef pokitt::Enthalpy<FieldT>::Builder EnthalpyIC;
      icFactory.register_expression( scinew EnthalpyIC( primVarInitTag_,
                                                        tags.temperature,
                                                        yiInitTags_ ) );

      typedef ExprAlgebra<FieldT> ExprAlgbr;
      return icFactory.register_expression( new ExprAlgbr::Builder( this->initial_condition_tag(),
                                                                    tag_list( primVarInitTag_, densityInitTag_ ),
                                                                    ExprAlgbr::PRODUCT ) );
    }
    else
#endif
    {
      return ScalarTransportEquation<FieldT>::initial_condition(icFactory);
    }

  }
  //---------------------------------------------------------------------------

} /* namespace WasatchCore */
