/*
 * The MIT License
 *
 * Copyright (c) 2015-2018 The University of Utah
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

#include <Core/Exceptions/ProblemSetupException.h>

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/Transport/CompressibleMomentumTransportEquation.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/MomentumRHS.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OutflowBC.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/OpenBC.h>
#include <CCA/Components/Wasatch/Expressions/PostProcessing/Derivative.h>

#ifdef HAVE_POKITT
#include <pokitt/CanteraObjects.h>
#include <pokitt/MixtureMolWeight.h>
#include <pokitt/thermo/Pressure.h>
#endif

// -- NSCBC Includes -- //
#include <nscbc/CharacteristicBCBuilder.h>
#include <nscbc/TagManager.h>
#include <nscbc/SpeedOfSound.h>



namespace WasatchCore{


  //====================================================================
  // Find out if a certain momentum component of MomDirT is perpendicular to face
  template <typename MomDirT> bool is_perpendicular(const Uintah::Patch::FaceType face)
  {return false;}
  
  template<>
  bool is_perpendicular<SpatialOps::XDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::xminus || face == Uintah::Patch::xplus) {
      return true;
    } else {
      return false;
    }
  }

  template<>
  bool is_perpendicular<SpatialOps::YDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::yminus || face == Uintah::Patch::yplus) {
      return true;
    } else {
      return false;
    }
  }

  template<>
  bool is_perpendicular<SpatialOps::ZDIR>(const Uintah::Patch::FaceType face)
  {
    if (face == Uintah::Patch::zminus || face == Uintah::Patch::zplus) {
      return true;
    } else {
      return false;
    }
  }

  //====================================================================
  // Typedef the normal direction based on DirT
  template< typename DirT>
  struct NormalDirTypeSelector
  {
  public:
    typedef typename SpatialOps::SSurfXField   NormalFaceT;
    typedef typename SpatialOps::XDIR          NormalDirT;
  };

  template<>
  struct NormalDirTypeSelector<SpatialOps::XDIR>
  {
    typedef typename SpatialOps::SSurfXField   NormalFaceT;
    typedef typename SpatialOps::XDIR          NormalDirT;
  };
  
  template<>
  struct NormalDirTypeSelector<SpatialOps::YDIR>
  {
    typedef typename SpatialOps::SSurfYField   NormalFaceT;
    typedef typename SpatialOps::YDIR          NormalDirT;
  };

  template<>
  struct NormalDirTypeSelector<SpatialOps::ZDIR>
  {
    typedef typename SpatialOps::SSurfZField   NormalFaceT;
    typedef typename SpatialOps::ZDIR          NormalDirT;
  };

  //============================================================================
  // Typedef the shear directions based on DirT and the XYZ convention
  template< typename MomentumDirT> struct StrainDirTypeSelector;

  // for tau_x
  // normal: tau_xx
  // shear1: tau_xy
  // shear2: tau_xz
  template<> struct StrainDirTypeSelector<SpatialOps::XDIR>
  {
  public:
    typedef SpatialOps::SSurfYField Strain1FaceT;
    typedef SpatialOps::SSurfZField Strain2FaceT;
    typedef SpatialOps::YDIR        Strain1DirT;
    typedef SpatialOps::ZDIR        Strain2DirT;
  };

  // for tau_y
  // normal: tau_yy
  // shear1: tau_yz
  // shear2: tau_yx
  template<> struct StrainDirTypeSelector<SpatialOps::YDIR>
  {
  public:
    typedef SpatialOps::SSurfZField Strain1FaceT;
    typedef SpatialOps::SSurfXField Strain2FaceT;
    typedef SpatialOps::ZDIR        Strain1DirT;
    typedef SpatialOps::XDIR        Strain2DirT;
  };

  // for tau_z
  // normal: tau_zz
  // shear1: tau_zx
  // shear2: tau_zy
  template<> struct StrainDirTypeSelector<SpatialOps::ZDIR>
  {
  public:
    typedef SpatialOps::SSurfXField Strain1FaceT;
    typedef SpatialOps::SSurfYField Strain2FaceT;
    typedef SpatialOps::XDIR        Strain1DirT;
    typedef SpatialOps::YDIR        Strain2DirT;
  };
  //============================================================================
  

  template< typename MomDirT, typename FaceT, typename GradT >
  struct CompressibleMomentumBoundaryTyper
  {
    typedef SpatialOps::SVolField CellT;
    typedef SpatialOps::Divergence DivT;
    typedef SpatialOps::Interpolant InterpT;

    typedef typename NormalDirTypeSelector<MomDirT>::NormalFaceT NormalFluxT;
    typedef typename StrainDirTypeSelector<MomDirT>::Strain1FaceT Transverse1FluxT;
    typedef typename StrainDirTypeSelector<MomDirT>::Strain2FaceT Transverse2FluxT;

    typedef typename SpatialOps::OperatorTypeBuilder<InterpT, CellT, FaceT>::type CellDirichletT;
    typedef typename SpatialOps::OperatorTypeBuilder<GradT, CellT, CellT>::type CellNeumannT;
    typedef typename SpatialOps::OperatorTypeBuilder<DivT, NormalFluxT, CellT>::type NormalFaceNeumannT;
    typedef typename SpatialOps::OperatorTypeBuilder<DivT, Transverse1FluxT, CellT>::type Transverse1FaceNeumannT;
    typedef typename SpatialOps::OperatorTypeBuilder<DivT, Transverse2FluxT, CellT>::type Transverse2FaceNeumannT;

    typedef typename SpatialOps::NeboBoundaryConditionBuilder<CellDirichletT> CellDirichletBCOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<CellNeumannT> CellNeumannBCOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<NormalFaceNeumannT> NormalFaceNeumannBCOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<Transverse1FaceNeumannT> Transverse1FaceNeumannBCOpT;
    typedef typename SpatialOps::NeboBoundaryConditionBuilder<Transverse2FaceNeumannT> Transverse2FaceNeumannBCOpT;

    typedef typename ConstantBCNew<CellT,CellDirichletBCOpT>::Builder ConstantCellDirichletBC;
    typedef typename ConstantBCNew<CellT,CellNeumannBCOpT>::Builder ConstantCellNeumannBC;
    typedef typename VelocityDependentConstantBC<CellT,CellDirichletBCOpT>::Builder ConstantVelDepCellDirichletBC;
    typedef typename VelocityDependentConstantBC<CellT,CellNeumannBCOpT>::Builder ConstantVelDepCellNeumannBC;
    typedef typename ConstantVelDepCellDirichletBC::FlowType CellDirichletBCFlowTypeT;
    typedef typename ConstantVelDepCellNeumannBC::FlowType CellNeumannBCFlowTypeT;
    const CellDirichletBCFlowTypeT CELL_DIRICHLET_ON_INFLOW = CellDirichletBCFlowTypeT::APPLY_ON_INFLOW;
    const CellDirichletBCFlowTypeT CELL_DIRICHLET_ON_OUTFLOW = CellDirichletBCFlowTypeT::APPLY_ON_OUTFLOW;
    const CellNeumannBCFlowTypeT CELL_NEUMANN_ON_INFLOW = CellNeumannBCFlowTypeT::APPLY_ON_INFLOW;
    const CellNeumannBCFlowTypeT CELL_NEUMANN_ON_OUTFLOW = CellNeumannBCFlowTypeT::APPLY_ON_OUTFLOW;

    typedef typename ConstantBCNew<NormalFluxT,NormalFaceNeumannBCOpT>::Builder ConstantNormalFaceNeumannBC;
    typedef typename VelocityDependentConstantBC<NormalFluxT,NormalFaceNeumannBCOpT>::Builder ConstantVelDepNormalFaceNeumannBC;
    typedef typename ConstantVelDepNormalFaceNeumannBC::FlowType NormalFaceNeumannBCFlowTypeT;
    const NormalFaceNeumannBCFlowTypeT NORMAL_FLUX_NEUMANN_ON_INFLOW = NormalFaceNeumannBCFlowTypeT::APPLY_ON_INFLOW;
    const NormalFaceNeumannBCFlowTypeT NORMAL_FLUX_NEUMANN_ON_OUTFLOW = NormalFaceNeumannBCFlowTypeT::APPLY_ON_OUTFLOW;

    typedef typename ConstantBCNew<Transverse1FluxT,Transverse1FaceNeumannBCOpT>::Builder ConstantTransverse1FaceNeumannBC;
    typedef typename VelocityDependentConstantBC<Transverse1FluxT,Transverse1FaceNeumannBCOpT>::Builder ConstantVelDepTransverse1FaceNeumannBC;
    typedef typename ConstantVelDepTransverse1FaceNeumannBC::FlowType Transverse1FaceNeumannBCFlowTypeT;
    const Transverse1FaceNeumannBCFlowTypeT TRANSVERSE1_FLUX_NEUMANN_ON_INFLOW = Transverse1FaceNeumannBCFlowTypeT::APPLY_ON_INFLOW;
    const Transverse1FaceNeumannBCFlowTypeT TRANSVERSE1_FLUX_NEUMANN_ON_OUTFLOW = Transverse1FaceNeumannBCFlowTypeT::APPLY_ON_OUTFLOW;

    typedef typename ConstantBCNew<Transverse2FluxT,Transverse2FaceNeumannBCOpT>::Builder ConstantTransverse2FaceNeumannBC;
    typedef typename VelocityDependentConstantBC<Transverse2FluxT,Transverse2FaceNeumannBCOpT>::Builder ConstantVelDepTransverse2FaceNeumannBC;
    typedef typename ConstantVelDepTransverse2FaceNeumannBC::FlowType Transverse2FaceNeumannBCFlowTypeT;
    const Transverse2FaceNeumannBCFlowTypeT TRANSVERSE2_FLUX_NEUMANN_ON_INFLOW = Transverse2FaceNeumannBCFlowTypeT::APPLY_ON_INFLOW;
    const Transverse2FaceNeumannBCFlowTypeT TRANSVERSE2_FLUX_NEUMANN_ON_OUTFLOW = Transverse2FaceNeumannBCFlowTypeT::APPLY_ON_OUTFLOW;
  };

  /**
   *  \class IdealGasPressure
   *  \author James C. Sutherland
   *  \date November, 2015
   *
   *  \brief Calculates the pressure from the ideal gas law: \f$p=\frac{\rho R T}{M}\f$
   *   where \f$M\f$ is the mixture molecular weight.
   */
  template< typename FieldT >
  class IdealGasPressure : public Expr::Expression<FieldT>
  {
    double gasConstant_;
    DECLARE_FIELDS( FieldT, density_, temperature_, mixMW_ )
    
    IdealGasPressure( const Expr::Tag& densityTag,
                      const Expr::Tag& temperatureTag,
                      const Expr::Tag& mixMWTag,
                      Uintah::ProblemSpecP wasatchSpec )
    : Expr::Expression<FieldT>(),
      gasConstant_( 8314.459848 )  // gas constant J/(kmol K)
    {
      this->set_gpu_runnable(true);
      density_     = this->template create_field_request<FieldT>( densityTag     );
      temperature_ = this->template create_field_request<FieldT>( temperatureTag );
      mixMW_       = this->template create_field_request<FieldT>( mixMWTag       );

#ifdef HAVE_POKITT
      Uintah::ProblemSpecP speciesParams = wasatchSpec->findBlock("SpeciesTransportEquations");
      if( speciesParams ){
        gasConstant_ = CanteraObjects::gas_constant();
      }
#endif
    }
    
  public:
    
    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag densityTag_, temperatureTag_, mixMWTag_;
      Uintah::ProblemSpecP wasatchSpec_;
    public:
      /**
       *  @brief Build a IdealGasPressure expression
       *  @param resultTag the tag for the value that this expression computes
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& densityTag,
               const Expr::Tag& temperatureTag,
               const Expr::Tag& mixMWTag,
               Uintah::ProblemSpecP wasatchSpec,
               const int nghost = DEFAULT_NUMBER_OF_GHOSTS )
      : ExpressionBuilder( resultTag, nghost ),
        densityTag_( densityTag ),
        temperatureTag_( temperatureTag ),
        mixMWTag_( mixMWTag ),
        wasatchSpec_( wasatchSpec )
      {}
      
      Expr::ExpressionBase* build() const{
        return new IdealGasPressure<FieldT>( densityTag_, temperatureTag_, mixMWTag_, wasatchSpec_ );
      }
      
    };  /* end of Builder class */
    
    ~IdealGasPressure(){}
    
    void evaluate()
    {
      FieldT& result = this->value();
      const FieldT& density     = density_    ->field_ref();
      const FieldT& temperature = temperature_->field_ref();
      const FieldT& mixMW       = mixMW_      ->field_ref();
      result <<= density * gasConstant_ * temperature / mixMW;
    }
  };
  
  //============================================================================
  
  Expr::ExpressionID
  ContinuityTransportEquation::initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    typedef Density_IC<MyFieldT>::Builder DensIC;
    return exprFactory.register_expression( scinew DensIC( initial_condition_tag(),
                                                           temperatureTag_,
                                                           TagNames::self().pressure,
                                                           mixMWTag_,
                                                           wasatchSpec_ ) );
  }

  //============================================================================

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
  CompressibleMomentumTransportEquation( Uintah::ProblemSpecP wasatchSpec,
                                         const Direction momComponent,
                                         const std::string velName,
                                         const std::string momName,
                                         const Expr::Tag densityTag,
                                         const Expr::Tag temperatureTag,
                                         const Expr::Tag mixMWTag,
                                         const Expr::Tag e0Tag, // total internal energy tag
                                         const Expr::Tag bodyForceTag,
                                         const Expr::Tag srcTermTag,
                                         GraphCategories& gc,
                                         Uintah::ProblemSpecP params,
                                         TurbulenceParameters turbParams )
  : MomentumTransportEquationBase<SVolField>( momComponent,
                                              velName,
                                              momName,
                                              densityTag,
                                              bodyForceTag,
                                              srcTermTag,
                                              gc,
                                              params,
                                              turbParams ),
   wasatchSpec_( wasatchSpec ),
   temperatureTag_(temperatureTag),
   mixMWTag_(mixMWTag),
   e0Tag_(e0Tag)
  {
    // todo:
    //  - strain tensor       // registered by MomentumTransportEquationBase
    //  - convective flux     // registered by the MomentumTransportEquationBase
    //  - turbulent viscosity // SHOULD be registered by the MomentumTransportEquationBase. NOT READY YET.
    //  - buoyancy? //        // body forces are handled in the momentumequationbase

    Expr::ExpressionFactory& factory = *gc[ADVANCE_SOLUTION]->exprFactory;

    Uintah::ProblemSpecP speciesParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
    if( speciesParams ){
#ifdef HAVE_POKITT
      typedef pokitt::Pressure<FieldT>::Builder Pressure;
      if( !factory.have_entry(TagNames::self().pressure) ){
        const Expr::ExpressionID pid = factory.register_expression( scinew Pressure( TagNames::self().pressure,
                                                                                     temperatureTag,
                                                                                     densityTag,
                                                                                     mixMWTag ) );
        factory.cleave_from_parents( pid );
      }
#endif
    }
    else{
      typedef IdealGasPressure<FieldT>::Builder Pressure;
      if( !factory.have_entry(TagNames::self().pressure) ){
        const Expr::ExpressionID pid = factory.register_expression( scinew Pressure( TagNames::self().pressure,
                                                                                     densityTag,
                                                                                     temperatureTag,
                                                                                     mixMWTag,
                                                                                     wasatchSpec_ ) );
        factory.cleave_from_parents( pid );
      }
    }

    setup();
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  Expr::ExpressionID
  CompressibleMomentumTransportEquation<MomDirT>::
  setup_rhs( FieldTagInfo&,
             const Expr::TagList& srcTags )
  {
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    Expr::Tag volFracTag = vNames.vol_frac_tag<FieldT>();
    
    Expr::ExpressionFactory& factory = *this->gc_[ADVANCE_SOLUTION]->exprFactory;

    typedef typename MomRHS<SVolField, MomDirT>::Builder RHS;
    return factory.register_expression( scinew RHS( this->rhsTag_,
                                                    this->pressureTag_,
                                                    rhs_part_tag(this->solnVarTag_),
                                                    volFracTag ) );
  }

  //----------------------------------------------------------------------------

  template <typename MomDirT>
  CompressibleMomentumTransportEquation<MomDirT>::
  ~CompressibleMomentumTransportEquation()
  {}

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  setup_boundary_conditions( WasatchBCHelper& bcHelper,
                             GraphCategories& graphCat )
  {
    Expr::ExpressionFactory& advSlnFactory = *(graphCat[ADVANCE_SOLUTION]->exprFactory);
    Expr::ExpressionFactory& initFactory   = *(graphCat[INITIALIZATION  ]->exprFactory);
    
    const TagNames& tagNames = TagNames::self();
    //
    // Add dummy modifiers on all patches. This is used to inject new dependencies across all patches.
    // Those new dependencies result for example from complicated boundary conditions added in this
    // function. NOTE: whenever you want to add a new complex boundary condition, please use this
    // functionality to inject new dependencies across patches.
    //
    {
      // add dt dummy modifier for outflow bcs...
      bcHelper.create_dummy_dependency<SpatialOps::SingleValueField, FieldT>(rhs_part_tag(this->solnVarName_), tag_list(tagNames.dt), ADVANCE_SOLUTION);
      
      // add momentum dummy modifiers
      const Expr::Tag momTimeAdvanceTag(this->solnvar_np1_tag());
      bcHelper.create_dummy_dependency<SVolField, FieldT>(momTimeAdvanceTag, tag_list(this->densityTag_),ADVANCE_SOLUTION);
      bcHelper.create_dummy_dependency<FieldT, FieldT>(momTimeAdvanceTag, tag_list(this->thisVelTag_),ADVANCE_SOLUTION);
      if( initFactory.have_entry(this->thisVelTag_) ){
        const Expr::Tag densityStateNone(this->densityTag_.name(), Expr::STATE_NONE);
        bcHelper.create_dummy_dependency<SVolField, FieldT>(momTimeAdvanceTag, tag_list(densityStateNone),INITIALIZATION);
        bcHelper.create_dummy_dependency<FieldT, FieldT>(momTimeAdvanceTag, tag_list(this->thisVelTag_),INITIALIZATION);
      }
    }
    //
    // END DUMMY MODIFIER SETUP
    //
    
    // make logical decisions based on the specified boundary types
    int jobid = 0;

    // we will need the speed of sound
    if( !( advSlnFactory.have_entry(tagNames.soundspeed) ) ){
      typedef typename NSCBC::SpeedOfSound<SVolField>::Builder SoundSpeed;
      advSlnFactory.register_expression( new SoundSpeed( tagNames.soundspeed, this->pressureTag_, this->densityTag_, tagNames.cp, tagNames.cv ) );
    }

    // get reference pressure
    double refPressure = 101325.0;
    if (wasatchSpec_->findBlock("NSCBC")) {
      Uintah::ProblemSpecP nscbcXMLSpec = wasatchSpec_->findBlock("NSCBC");
      nscbcXMLSpec->getAttribute("pfarfield", refPressure);
    }

    // set up the extra fields for setting BCs on primitive and conserved variables
    const Expr::Tag temporaryUTag( "temporary_" + this->thisVelTag_.name() + "_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryPTag( "temporary_pressure_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryRhoTag( "temporary_rho_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryRhoUTag( "temporary_rho" + this->thisVelTag_.name() + "_for_bcs", Expr::STATE_NONE );

    if( !( advSlnFactory.have_entry( temporaryUTag ) ) ){
      advSlnFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryUTag, 0.0 ) );
    }
    if( !( initFactory.have_entry( temporaryUTag ) ) ){
      initFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryUTag, 0.0 ) );
    }
    if( !( advSlnFactory.have_entry( temporaryPTag ) ) ){
      advSlnFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryPTag, refPressure ) );
    }
    if( !( initFactory.have_entry( temporaryPTag ) ) ){
      initFactory.register_expression( new Expr::ConstantExpr<SVolField>::Builder( temporaryPTag, refPressure ) );
    }
    if( !( advSlnFactory.have_entry( temporaryRhoUTag ) ) ){
      typedef ExprAlgebra<SVolField>::Builder RhoU;
      advSlnFactory.register_expression( new RhoU( temporaryRhoUTag, Expr::tag_list( temporaryRhoTag, temporaryUTag ), ExprAlgebra<SVolField>::PRODUCT ) );
    }
    if( !( initFactory.have_entry( temporaryRhoUTag ) ) ){
      typedef ExprAlgebra<SVolField>::Builder RhoU;
      initFactory.register_expression( new RhoU( temporaryRhoUTag, Expr::tag_list( temporaryRhoTag, temporaryUTag ), ExprAlgebra<SVolField>::PRODUCT ) );
    }


    BOOST_FOREACH( const BndMapT::value_type& bndPair, bcHelper.get_boundary_information() )
    {
      const std::string& bndName = bndPair.first;
      const BndSpec& myBndSpec = bndPair.second;

      // a lambda to make decorated tags for boundary condition expressions
      //
      // param: exprTag: the Expr::Tag for the field on which we will impose boundary conditions
      // param: description: a string describing the boundary condition, such as "neumann-zero-for-outflow" or "dirichlet-for-inflow"
      // param: direction: a string for the direction of the boundary face, such as "X", "Y", or "Z"
      auto get_decorated_tag = [&myBndSpec]( const Expr::Tag exprTag, const std::string description, const std::string direction) -> Expr::Tag
      {
        return Expr::Tag( exprTag.name() + "_STATE_NONE_" + description + "_bc_" + myBndSpec.name + "_" + direction + "dir", Expr::STATE_NONE );
      };
      
//      const bool isNormal = is_normal_to_boundary(this->staggered_location(), myBndSpec.face);

      //============================================================================================
      // NSCBC TREATMENT
      std::map< NSCBC::TagName, Expr::Tag > tags;
      std::map< NSCBC::TagListName, Expr::TagList > tagLists;
      {
        int normVelIndex = 0; // velocity component normal to this boundary face u_i. This will be picked from velTags_[]
        switch (myBndSpec.face) {
          case Uintah::Patch::xminus:
          case Uintah::Patch::xplus:
            normVelIndex = 0;
            break;
          case Uintah::Patch::yminus:
          case Uintah::Patch::yplus:
            normVelIndex = 1;
            break;
          case Uintah::Patch::zminus:
          case Uintah::Patch::zplus:
            normVelIndex = 2;
            break;
          default:
            break;
        }
        
        // we will need dudx, dvdy, and dwdz. Register here if they were not registered
        const Expr::TagList dVelTags = tag_list(tagNames.dudx, tagNames.dvdy, tagNames.dwdz);
        if( !( advSlnFactory.have_entry(tagNames.dudx) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::XDIR>::Builder dudx;
          advSlnFactory.register_expression(new dudx( tagNames.dudx, this->velTags_[0] ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dvdy) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::YDIR>::Builder dvdy;
          advSlnFactory.register_expression(new dvdy( tagNames.dvdy, this->velTags_[1] ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dwdz) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::ZDIR>::Builder dwdz;
          advSlnFactory.register_expression(new dwdz( tagNames.dwdz, this->velTags_[2] ));
        }
        
        // we will also need dpdx, dpdy, and dpdz. Register here if they were not registered
        const Expr::TagList dPTags = tag_list(tagNames.dpdx, tagNames.dpdy,tagNames.dpdz);
        if( !( advSlnFactory.have_entry(tagNames.dpdx) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::XDIR>::Builder dpdx;
          advSlnFactory.register_expression(new dpdx( tagNames.dpdx, this->pressureTag_ ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dpdy) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::YDIR>::Builder dpdy;
          advSlnFactory.register_expression(new dpdy( tagNames.dpdy, this->pressureTag_ ));
        }
        
        if( !( advSlnFactory.have_entry(tagNames.dpdz) ) ){
          typedef typename Derivative<SVolField, SVolField, SpatialOps::ZDIR>::Builder dpdz;
          advSlnFactory.register_expression(new dpdz( tagNames.dpdz, this->pressureTag_ ));
        }
        
        // now fill in the tags for the NSCBC tag manager
        tags[NSCBC::U]       = this->velTags_[0];
        tags[NSCBC::V]       = this->velTags_[1];
        tags[NSCBC::W]       = this->velTags_[2];
        tags[NSCBC::FACEVEL] = this->velTags_[normVelIndex];
        tags[NSCBC::DVELHARDINFLOW]    = dVelTags[normVelIndex];
        tags[NSCBC::T]       = this->temperatureTag_;
        tags[NSCBC::P]       = this->pressureTag_;
        tags[NSCBC::DPHARDINFLOW]   = dPTags[normVelIndex];
        tags[NSCBC::RHO]     = this->densityTag_;
        tags[NSCBC::MMW]     = this->mixMWTag_;
        tags[NSCBC::CP]      = tagNames.cp;
        tags[NSCBC::CV]      = tagNames.cv;
        tags[NSCBC::C]       = tagNames.soundspeed;
        tags[NSCBC::E0]      = this->e0Tag_;

        bool doSpecies = false;
#ifdef HAVE_POKITT
        Uintah::ProblemSpecP speciesParams = wasatchSpec_->findBlock("SpeciesTransportEquations");
        if( speciesParams ){
          doSpecies = true;
          Expr::TagList yiTags, riTags, hiTags;
          for( int i=0; i<CanteraObjects::number_species(); ++i ){
            const std::string specName = CanteraObjects::species_name(i);
            const Expr::Tag specTag( specName, Expr::STATE_NONE );
            yiTags.push_back( specTag );
            const Expr::Tag rateTag( "rr_" + specName, Expr::STATE_NONE );
            riTags.push_back( rateTag );
            const Expr::Tag enthalpyTag( specName + "_" + tagNames.enthalpy.name(), Expr::STATE_NONE );
            hiTags.push_back( enthalpyTag );
          }
          tagLists[NSCBC::Y_N] = yiTags;
          tagLists[NSCBC::R_N] = riTags;
          tagLists[NSCBC::H_N] = hiTags;
        }
#endif
        if( !doSpecies ){
          tagLists[NSCBC::H_N] = tag_list(tagNames.enthalpy);
        }

        // create the NSCBC tag manager
        NSCBC::TagManager nscbcTagMgr( tags, tagLists, doSpecies );
        
        bcHelper.setup_nscbc<MomDirT>(myBndSpec, nscbcTagMgr, jobid++);
      }
      //============================================================================================

      switch (myBndSpec.type) {
        case WALL:
        {
          // first check if the user specified momentum boundary conditions at the wall
          if( myBndSpec.has_field(this->thisVelTag_.name()) || myBndSpec.has_field(this->solnVarName_) ||
             myBndSpec.has_field(this->rhs_name()) || myBndSpec.has_field(this->solnVarName_ + "_rhs_part") ){
            std::ostringstream msg;
            msg << "ERROR: You cannot specify any momentum-related boundary conditions at a stationary wall. "
            << "This error occured while trying to analyze boundary " << bndName
            << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          }
          
          // set zero momentum on the wall
          BndCondSpec momBCSpec = {this->solution_variable_name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momBCSpec);

          // set zero pressure gradient
          BndCondSpec pressureBCSpec = {this->pressureTag_.name(),"none" ,0.0, NEUMANN, DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, pressureBCSpec);
          
          // set zero velocity
          BndCondSpec velBCSpec = {this->thisVelTag_.name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, velBCSpec);
          
          // set zero momentum RHS on the wall
          BndCondSpec momRHSBCSpec = {this->rhs_tag().name(),"none" ,0.0,DIRICHLET,DOUBLE_TYPE};
          bcHelper.add_boundary_condition(bndName, momRHSBCSpec);

          break;
        }
        case VELOCITY: // we will use this as the hard inflow
        {
          // one-sided pressure
          typedef typename SpatialOps::UnitTriplet<MomDirT>::type UnitTripletT;

          Expr::Tag oneSidedPressureCorrectionTag;
          Expr::ExpressionBuilder* oneSidedPressureCorrectionBuilder = NULL;

          switch (myBndSpec.face) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::yplus:
            case Uintah::Patch::zplus:
            {
              if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<typename UnitTripletT::Negate>,FieldT>::type OpT;
                oneSidedPressureCorrectionTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_plus_side_" + bndName, Expr::STATE_NONE);
                typedef typename BCOneSidedGradP<FieldT,OpT>::Builder pbuilderT;
                oneSidedPressureCorrectionBuilder = new pbuilderT( oneSidedPressureCorrectionTag, this->pressureTag_ );
              }
            }
            break;
            case Uintah::Patch::xminus:
            case Uintah::Patch::yminus:
            case Uintah::Patch::zminus:
            {
              if ( is_perpendicular<MomDirT>(myBndSpec.face) ) {
                typedef typename SpatialOps::OneSidedOpTypeBuilder<SpatialOps::Gradient,SpatialOps::OneSidedStencil3<UnitTripletT>,FieldT>::type OpT;
                oneSidedPressureCorrectionTag = Expr::Tag(this->solnVarName_ + "_rhs_mod_minus_side_" + bndName, Expr::STATE_NONE);
                typedef typename BCOneSidedGradP<FieldT,OpT>::Builder pbuilderT;
                oneSidedPressureCorrectionBuilder = new pbuilderT( oneSidedPressureCorrectionTag, this->pressureTag_ );
              }
            }
            break;
            default:
              break;
          }

          if( is_perpendicular<MomDirT>( myBndSpec.face ) ) {
            advSlnFactory.register_expression( oneSidedPressureCorrectionBuilder );
          }

          // tags for modifier expressions and strings for BC spec names, will depend on boundary face
          Expr::Tag bcCopiedMomentumTag, neumannZeroNormalStrainTag, neumannZeroPressureTag;

          // build boundary conditions for x, y, and z faces
          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              const std::string dir = "X";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfXField, SpatialOps::GradientX> BCTypes;

              neumannZeroNormalStrainTag = get_decorated_tag( this->normalStrainTag_, "neumann-zero-hard-inflow", dir );
              bcCopiedMomentumTag        = get_decorated_tag( this->solnVarTag_     , "bccopy-hard-inflow"      , dir );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroNormalStrainTag, 0.0 ) );
              if( !initFactory  .have_entry( bcCopiedMomentumTag        ) ) initFactory  .register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
              if( !advSlnFactory.have_entry( bcCopiedMomentumTag        ) ) advSlnFactory.register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
            }
            break;
            case Uintah::Patch::yplus:
            case Uintah::Patch::yminus:
            {
              const std::string dir = "Y";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfYField, SpatialOps::GradientY> BCTypes;

              neumannZeroNormalStrainTag = get_decorated_tag( this->normalStrainTag_, "neumann-zero-for-inflow", dir );
              bcCopiedMomentumTag        = get_decorated_tag( this->solnVarTag_     , "bccopy-hard-inflow"      , dir );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroNormalStrainTag, 0.0 ) );
              if( !initFactory  .have_entry( bcCopiedMomentumTag        ) ) initFactory  .register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
              if( !advSlnFactory.have_entry( bcCopiedMomentumTag        ) ) advSlnFactory.register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
            }
            break;
            case Uintah::Patch::zplus:
            case Uintah::Patch::zminus:
            {
              const std::string dir = "Z";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfZField, SpatialOps::GradientZ> BCTypes;

              neumannZeroNormalStrainTag = get_decorated_tag( this->normalStrainTag_, "neumann-zero-hard-inflow", dir );
              bcCopiedMomentumTag        = get_decorated_tag( this->solnVarTag_     , "bccopy-hard-inflow"      , dir );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroNormalStrainTag, 0.0 ) );
              if( !initFactory  .have_entry( bcCopiedMomentumTag        ) ) initFactory  .register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
              if( !advSlnFactory.have_entry( bcCopiedMomentumTag        ) ) advSlnFactory.register_expression( new BCCopier<SVolField>::Builder( bcCopiedMomentumTag, temporaryRhoUTag ) );
            }
            break;
            default:
              break;
          }

          BndCondSpec momentumDirichletBC = {this->solnVarTag_.name(), bcCopiedMomentumTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
          bcHelper.add_boundary_condition( bndName, momentumDirichletBC );

          if( is_perpendicular<MomDirT>( myBndSpec.face ) ){
            BndCondSpec oneSidedPressureBC = {this->rhsTag_.name()         , oneSidedPressureCorrectionTag.name(), 0.0, DIRICHLET, FUNCTOR_TYPE};
            BndCondSpec stressBCSpecNormal = {this->normalStrainTag_.name(), neumannZeroNormalStrainTag.name()   , 0.0, NEUMANN  , FUNCTOR_TYPE};
            bcHelper.add_boundary_condition( bndName, stressBCSpecNormal  );
            bcHelper.add_boundary_condition( bndName, oneSidedPressureBC  );
          }

        }
        break;


        case OUTFLOW:
        case OPEN:
        {
          // check to do transverse strain BCs
          const bool doStrain1 = advSlnFactory.have_entry( this->shearStrainTag1_ );
          const bool doStrain2 = advSlnFactory.have_entry( this->shearStrainTag2_ );

          // tags for modifier expressions and strings for BC spec names, will depend on boundary face
          Expr::Tag neumannZeroConvectiveFluxTag,
                    neumannZeroNormalStrainTag,
                    neumannZeroShearStrain1Tag,
                    neumannZeroShearStrain2Tag,
                    nonreflectingPressureTag,
                    neumannZeroPressureTag,
                    neumannZeroMomentumTag;

          // build boundary conditions for x, y, and z faces
          switch( myBndSpec.face ) {
            case Uintah::Patch::xplus:
            case Uintah::Patch::xminus:
            {
              const std::string dir = "X";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfXField, SpatialOps::GradientX> BCTypes;
              BCTypes bcTypes;

              neumannZeroConvectiveFluxTag = get_decorated_tag( this->normalConvFluxTag_, "neumann-zero-nonreflecting", dir );
              neumannZeroPressureTag       = get_decorated_tag( this->pressureTag_      , "neumann-zero-nonreflecting", dir );
              neumannZeroMomentumTag       = get_decorated_tag( this->solnVarTag_       , "neumann-zero-nonreflecting", dir );
              neumannZeroNormalStrainTag   = get_decorated_tag( this->normalStrainTag_  , "neumann-zero-nonreflecting", dir );
              neumannZeroShearStrain1Tag   = get_decorated_tag( this->shearStrainTag1_  , "neumann-outflow-zero-nonreflecting", dir );
              neumannZeroShearStrain2Tag   = get_decorated_tag( this->shearStrainTag2_  , "neumann-outflow-zero-nonreflecting", dir );

              if( !initFactory  .have_entry( neumannZeroConvectiveFluxTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC           ( neumannZeroNormalStrainTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain1Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse1FaceNeumannBC( neumannZeroShearStrain1Tag, 0.0, myBndSpec.face, velTags_[0], bcTypes.TRANSVERSE1_FLUX_NEUMANN_ON_OUTFLOW ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain2Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse2FaceNeumannBC( neumannZeroShearStrain2Tag, 0.0, myBndSpec.face, velTags_[0], bcTypes.TRANSVERSE2_FLUX_NEUMANN_ON_OUTFLOW ) );

              if( !initFactory  .have_entry( neumannZeroPressureTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroPressureTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroMomentumTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroMomentumTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
            }
            break;
            case Uintah::Patch::yplus:
            case Uintah::Patch::yminus:
            {
              const std::string dir = "Y";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfYField, SpatialOps::GradientY> BCTypes;
              BCTypes bcTypes;

              neumannZeroConvectiveFluxTag = get_decorated_tag( this->normalConvFluxTag_, "neumann-zero-nonreflecting", dir );
              neumannZeroPressureTag       = get_decorated_tag( this->pressureTag_      , "neumann-zero-nonreflecting", dir );
              neumannZeroMomentumTag       = get_decorated_tag( this->solnVarTag_       , "neumann-zero-nonreflecting", dir );
              neumannZeroNormalStrainTag   = get_decorated_tag( this->normalStrainTag_  , "neumann-zero-nonreflecting", dir );
              neumannZeroShearStrain1Tag   = get_decorated_tag( this->shearStrainTag1_  , "neumann-outflow-zero-nonreflecting", dir );
              neumannZeroShearStrain2Tag   = get_decorated_tag( this->shearStrainTag2_  , "neumann-outflow-zero-nonreflecting", dir );

              if( !initFactory  .have_entry( neumannZeroConvectiveFluxTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC           ( neumannZeroNormalStrainTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain1Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse1FaceNeumannBC( neumannZeroShearStrain1Tag, 0.0, myBndSpec.face, velTags_[1], bcTypes.TRANSVERSE1_FLUX_NEUMANN_ON_OUTFLOW ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain2Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse2FaceNeumannBC( neumannZeroShearStrain2Tag, 0.0, myBndSpec.face, velTags_[1], bcTypes.TRANSVERSE2_FLUX_NEUMANN_ON_OUTFLOW ) );

              if( !initFactory  .have_entry( neumannZeroPressureTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroPressureTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroMomentumTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroMomentumTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
            }
            break;
            case Uintah::Patch::zplus:
            case Uintah::Patch::zminus:
            {
              const std::string dir = "Z";
              typedef CompressibleMomentumBoundaryTyper<MomDirT, SpatialOps::SSurfZField, SpatialOps::GradientZ> BCTypes;
              BCTypes bcTypes;

              neumannZeroConvectiveFluxTag = get_decorated_tag( this->normalConvFluxTag_, "neumann-zero-nonreflecting", dir );
              neumannZeroPressureTag       = get_decorated_tag( this->pressureTag_      , "neumann-zero-nonreflecting", dir );
              neumannZeroMomentumTag       = get_decorated_tag( this->solnVarTag_       , "neumann-zero-nonreflecting", dir );
              neumannZeroNormalStrainTag   = get_decorated_tag( this->normalStrainTag_  , "neumann-zero-nonreflecting", dir );
              neumannZeroShearStrain1Tag   = get_decorated_tag( this->shearStrainTag1_  , "neumann-outflow-zero-nonreflecting", dir );
              neumannZeroShearStrain2Tag   = get_decorated_tag( this->shearStrainTag2_  , "neumann-outflow-zero-nonreflecting", dir );

              if( !initFactory  .have_entry( neumannZeroConvectiveFluxTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroConvectiveFluxTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC( neumannZeroConvectiveFluxTag, 0.0 ) );

              if( !advSlnFactory.have_entry( neumannZeroNormalStrainTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantNormalFaceNeumannBC           ( neumannZeroNormalStrainTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain1Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse1FaceNeumannBC( neumannZeroShearStrain1Tag, 0.0, myBndSpec.face, velTags_[2], bcTypes.TRANSVERSE1_FLUX_NEUMANN_ON_OUTFLOW ) );
              if( !advSlnFactory.have_entry( neumannZeroShearStrain2Tag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantVelDepTransverse2FaceNeumannBC( neumannZeroShearStrain2Tag, 0.0, myBndSpec.face, velTags_[2], bcTypes.TRANSVERSE2_FLUX_NEUMANN_ON_OUTFLOW ) );

              if( !initFactory  .have_entry( neumannZeroPressureTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroPressureTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroPressureTag, 0.0 ) );
              if( !initFactory  .have_entry( neumannZeroMomentumTag ) ) initFactory  .register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
              if( !advSlnFactory.have_entry( neumannZeroMomentumTag ) ) advSlnFactory.register_expression( new typename BCTypes::ConstantCellNeumannBC( neumannZeroMomentumTag, 0.0 ) );
            }
            break;
            default:
              break;
          }

          BndCondSpec momentumBCSpec = {this->solnVarName_             , neumannZeroMomentumTag.name()      , 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec pressureBCSpec = {this->pressureTag_.name()      , neumannZeroPressureTag.name()      , 0.0, NEUMANN, FUNCTOR_TYPE};
          BndCondSpec convFluxBCSpec = {this->normalConvFluxTag_.name(), neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};

          bcHelper.add_boundary_condition( bndName, pressureBCSpec );
          bcHelper.add_boundary_condition( bndName, momentumBCSpec );

          if( is_perpendicular<typename NormalDirTypeSelector<MomDirT>::NormalDirT>( myBndSpec.face ) ){
            // x-mom: if on x-face
            // y-mom: if on y-face
            // z-mom: if on z-face
            BndCondSpec convFluxBCSpec     = {this->normalConvFluxTag_.name(), neumannZeroConvectiveFluxTag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
            BndCondSpec stressBCSpecNormal = {this->normalStrainTag_.name()  , neumannZeroNormalStrainTag.name()  , 0.0, NEUMANN, FUNCTOR_TYPE};
            bcHelper.add_boundary_condition( bndName, convFluxBCSpec     );
            bcHelper.add_boundary_condition( bndName, stressBCSpecNormal );
          }
          else{
            if( doStrain1 ){
              if( is_perpendicular<typename StrainDirTypeSelector<MomDirT>::Strain1DirT>( myBndSpec.face ) ){
                // x-mom: if on y-face
                // y-mom: if on z-face
                // z-mom: if on x-face
                BndCondSpec stressBCSpecStrain1 = {this->shearStrainTag1_.name(), neumannZeroShearStrain1Tag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
                bcHelper.add_boundary_condition( bndName, stressBCSpecStrain1 );
              }
            }
            if( doStrain2 ){
              if( is_perpendicular<typename StrainDirTypeSelector<MomDirT>::Strain2DirT>( myBndSpec.face ) ){
                // x-mom: if on z-face
                // y-mom: if on x-face
                // z-mom: if on y-face
                BndCondSpec stressBCSpecStrain2 = {this->shearStrainTag2_.name(), neumannZeroShearStrain2Tag.name(), 0.0, NEUMANN, FUNCTOR_TYPE};
                bcHelper.add_boundary_condition( bndName, stressBCSpecStrain2 );
              }
            }
          }
        }
        break;

        case USER:
        default:
        {
          std::ostringstream msg;
          msg << "ERROR: VELOCITY, OPEN, and OUTFLOW boundary conditions are not currently supported for compressible flows in Wasatch. " << bndName
              << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
      } // SWITCH BOUNDARY TYPE
    } // BOUNDARY LOOP
  }

  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                     WasatchBCHelper& bcHelper )
  {
    const Category taskCat = INITIALIZATION;

    // apply velocity boundary condition, if specified
    bcHelper.apply_boundary_condition<FieldT>(this->thisVelTag_, taskCat, true);

    // tsaad: boundary conditions will not be applied on the initial condition of momentum. This leads
    // to tremendous complications in our graphs. Instead, specify velocity initial conditions
    // and velocity boundary conditions, and momentum bcs will appropriately propagate.
    bcHelper.apply_boundary_condition<FieldT>(this->initial_condition_tag(), taskCat, true);

    // bcs for hard inflow - set primitive and conserved variables
    const Expr::Tag temporaryUTag( "temporary_" + this->thisVelTag_.name() + "_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryPTag( "temporary_pressure_for_bcs", Expr::STATE_NONE );
    bcHelper.apply_boundary_condition<FieldT>( temporaryUTag, taskCat, true );
    bcHelper.apply_boundary_condition<FieldT>( temporaryPTag, taskCat, true );
  }
  
  //----------------------------------------------------------------------------

  template<typename MomDirT> NSCBC::TransportVal NSCBCMomentum();
  
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::XDIR>(){return NSCBC::MOMENTUM_X;}
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::YDIR>(){return NSCBC::MOMENTUM_Y;}
  template<> NSCBC::TransportVal NSCBCMomentum<SpatialOps::ZDIR>(){return NSCBC::MOMENTUM_Z;}
  
  //----------------------------------------------------------------------------
  
  template <typename MomDirT>
  void CompressibleMomentumTransportEquation<MomDirT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {
    const Category taskCat = ADVANCE_SOLUTION;
    // set bcs for momentum - use the TIMEADVANCE expression
    bcHelper.apply_boundary_condition<FieldT>( this->solnvar_np1_tag(), taskCat );
    // set bcs for velocity
    bcHelper.apply_boundary_condition<FieldT>( this->thisVelTag_, taskCat, true );
    // set bcs for partial rhs
    bcHelper.apply_boundary_condition<FieldT>( rhs_part_tag(mom_tag(this->solnVarName_)), taskCat, true);
    // set bcs for full rhs
    bcHelper.apply_boundary_condition<FieldT>( this->rhs_tag(), taskCat, true);
    // set bcs for pressure
    bcHelper.apply_boundary_condition<FieldT>( this->pressureTag_, taskCat);

    // bcs for hard inflow - set primitive and conserved variables
    const Expr::Tag temporaryUTag( "temporary_" + this->thisVelTag_.name() + "_for_bcs", Expr::STATE_NONE );
    const Expr::Tag temporaryPTag( "temporary_pressure_for_bcs", Expr::STATE_NONE );
    bcHelper.apply_boundary_condition<FieldT>( temporaryUTag, taskCat, true );
    bcHelper.apply_boundary_condition<FieldT>( temporaryPTag, taskCat, true );

    // set bcs for convective flux
    bcHelper.apply_boundary_condition< typename NormalDirTypeSelector<MomDirT>::NormalFaceT >( this->normalConvFluxTag_, taskCat);
    // set bcs for strain
    bcHelper.apply_boundary_condition< typename NormalDirTypeSelector<MomDirT>::NormalFaceT >( this->normalStrainTag_, taskCat);
    bcHelper.apply_boundary_condition< typename StrainDirTypeSelector<MomDirT>::Strain1FaceT >( this->shearStrainTag1_, taskCat);
    bcHelper.apply_boundary_condition< typename StrainDirTypeSelector<MomDirT>::Strain2FaceT >( this->shearStrainTag2_, taskCat);
    // apply NSCBC boundary conditions
    bcHelper.apply_nscbc_boundary_condition(this->rhs_tag(), NSCBCMomentum<MomDirT>(), taskCat);
  }

  //============================================================================
  
  template class CompressibleMomentumTransportEquation<SpatialOps::XDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::YDIR>;
  template class CompressibleMomentumTransportEquation<SpatialOps::ZDIR>;
} // namespace Wasatch



