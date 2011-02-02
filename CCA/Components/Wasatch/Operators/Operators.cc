//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggered.h>
// #include <spatialops/OperatorDatabase.h>
// #include <spatialops/structured/FVStaggeredFieldTypes.h>
// #include <spatialops/structured/FVStaggeredDivergence.h>
// #include <spatialops/structured/FVStaggeredGradient.h>
// #include <spatialops/structured/FVStaggeredInterpolant.h>
// #include <spatialops/structured/FVStaggeredScratch.h>
// #include <spatialops/structured/FVTopHatFilter.h>

//-- Wasatch includes --//
#include "Operators.h"
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

//-- Uintah includes --//
#include <Core/Grid/Patch.h>

using namespace SpatialOps;
using namespace structured;

namespace Wasatch{

  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB )
  {
    using namespace SpatialOps;
    using namespace structured;

    const SCIRun::IntVector udim = patch.getCellHighIndex() - patch.getCellLowIndex();
    std::vector<int> dim(3,1);
    for( size_t i=0; i<3; ++i ){ dim[i] = udim[i]; }

    const Uintah::Vector spacing = patch.dCell();
    std::vector<double> area(3,1);
    area[0] = spacing[1]*spacing[2];
    area[1] = spacing[0]*spacing[2];
    area[2] = spacing[0]*spacing[1];

    std::vector<bool> bcPlus(3,false);
    bcPlus[0] = patch.getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
    bcPlus[1] = patch.getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
    bcPlus[2] = patch.getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;

    const double vol = spacing[0]*spacing[1]*spacing[2];

    //--------------------------------------------------------
    // Divergence operators 
    //--------------------------------------------------------
    opDB.register_new_operator<DivSSurfXSVol>( DivSSurfXSVol::Assembler( dim, area[0], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivSSurfYSVol>( DivSSurfYSVol::Assembler( dim, area[1], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivSSurfZSVol>( DivSSurfZSVol::Assembler( dim, area[2], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<DivXSurfXXVol>( DivXSurfXXVol::Assembler( dim, area[0], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivXSurfYXVol>( DivXSurfYXVol::Assembler( dim, area[1], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivXSurfZXVol>( DivXSurfZXVol::Assembler( dim, area[2], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<DivYSurfXYVol>( DivYSurfXYVol::Assembler( dim, area[0], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivYSurfYYVol>( DivYSurfYYVol::Assembler( dim, area[1], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivYSurfZYVol>( DivYSurfZYVol::Assembler( dim, area[2], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<DivZSurfXZVol>( DivZSurfXZVol::Assembler( dim, area[0], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivZSurfYZVol>( DivZSurfYZVol::Assembler( dim, area[1], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<DivZSurfZZVol>( DivZSurfZZVol::Assembler( dim, area[2], vol, bcPlus[0], bcPlus[1], bcPlus[2] ) );


    //--------------------------------------------------------
    // gradient operators - diffusive fluxes
    //--------------------------------------------------------
    opDB.register_new_operator<GradSVolSSurfX>( GradSVolSSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradSVolSSurfY>( GradSVolSSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradSVolSSurfZ>( GradSVolSSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradXVolXSurfX>( GradXVolXSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradXVolXSurfY>( GradXVolXSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradXVolXSurfZ>( GradXVolXSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradYVolYSurfX>( GradYVolYSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradYVolYSurfY>( GradYVolYSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradYVolYSurfZ>( GradYVolYSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradZVolZSurfX>( GradZVolZSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradZVolZSurfY>( GradZVolZSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradZVolZSurfZ>( GradZVolZSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradSVolXVol>  ( GradSVolXVol  ::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );  
    opDB.register_new_operator<GradSVolYVol>  ( GradSVolYVol  ::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradSVolZVol>  ( GradSVolZVol  ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradXVolYSurfX>( GradXVolYSurfX::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradXVolZSurfX>( GradXVolZSurfX::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradYVolXSurfY>( GradYVolXSurfY::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradYVolZSurfY>( GradYVolZSurfY::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<GradZVolXSurfZ>( GradZVolXSurfZ::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradZVolYSurfZ>( GradZVolYSurfZ::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    // dilatation
    opDB.register_new_operator<GradXVolSVol>  ( GradXVolSVol  ::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradYVolSVol>  ( GradYVolSVol  ::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<GradZVolSVol>  ( GradZVolSVol  ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );


    //--------------------------------------------------------
    // interpolant scalar volume to scalar surface (diffusivities)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpSVolSSurfX>( InterpSVolSSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolSSurfY>( InterpSVolSSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolSSurfZ>( InterpSVolSSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );


    //--------------------------------------------------------
    // interpolants - scalar volume to staggered surfaces (viscosity, dilatation)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpSVolXSurfX>( InterpSVolXSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolXSurfY>( InterpSVolXSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolXSurfZ>( InterpSVolXSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<InterpSVolYSurfX>( InterpSVolYSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolYSurfY>( InterpSVolYSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolYSurfZ>( InterpSVolYSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    opDB.register_new_operator<InterpSVolZSurfX>( InterpSVolZSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolZSurfY>( InterpSVolZSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolZSurfZ>( InterpSVolZSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );


    //--------------------------------------------------------
    // interpolants - scalar volume to staggered volume (density)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpSVolXVol>( InterpSVolXVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolYVol>( InterpSVolYVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSVolZVol>( InterpSVolZVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );


    //--------------------------------------------------------
    // interpolants - staggered volume to staggered surfaces (advecting velocities)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpXVolYSurfX>( InterpXVolYSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpXVolZSurfX>( InterpXVolZSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                         
    opDB.register_new_operator<InterpYVolXSurfY>( InterpYVolXSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpYVolZSurfY>( InterpYVolZSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                         
    opDB.register_new_operator<InterpZVolXSurfZ>( InterpZVolXSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpZVolYSurfZ>( InterpZVolYSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    // interpolants - volume to surface for staggered cells.
    opDB.register_new_operator<InterpXVolXSurfX>( InterpXVolXSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpXVolXSurfY>( InterpXVolXSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpXVolXSurfZ>( InterpXVolXSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                           
    opDB.register_new_operator<InterpYVolYSurfX>( InterpYVolYSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpYVolYSurfY>( InterpYVolYSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpYVolYSurfZ>( InterpYVolYSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                           
    opDB.register_new_operator<InterpZVolZSurfX>( InterpZVolZSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpZVolZSurfY>( InterpZVolZSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpZVolZSurfZ>( InterpZVolZSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    //--------------------------------------------------------
    // interpolants - staggard volume to scalar surface  (advecting velocities)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpXVolSSurfX>( InterpXVolSSurfX::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpYVolSSurfY>( InterpYVolSSurfY::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpZVolSSurfZ>( InterpZVolSSurfZ::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

    //--------------------------------------------------------
    // UPWIND interpolants - phi volume to phi surface
    //--------------------------------------------------------    
    typedef UpwindInterpolant<SVolField,SSurfXField> UpwindSVolSSurfX;
    opDB.register_new_operator<UpwindSVolSSurfX>(scinew UpwindSVolSSurfX(dim, bcPlus) );
    
    typedef UpwindInterpolant<SVolField,SSurfYField> UpwindSVolSSurfY;
    opDB.register_new_operator<UpwindSVolSSurfY>(scinew UpwindSVolSSurfY(dim, bcPlus ));

    typedef UpwindInterpolant<SVolField,SSurfZField> UpwindSVolSSurfZ;
    opDB.register_new_operator<UpwindSVolSSurfZ>(scinew UpwindSVolSSurfZ(dim, bcPlus ));

    typedef UpwindInterpolant<XVolField,XSurfXField> UpwindXVolXSurfX;
    opDB.register_new_operator<UpwindXVolXSurfX>(scinew UpwindXVolXSurfX(dim, bcPlus ));
    
    typedef UpwindInterpolant<XVolField,XSurfYField> UpwindXVolXSurfY;
    opDB.register_new_operator<UpwindXVolXSurfY>(scinew UpwindXVolXSurfY(dim, bcPlus ));
    
    typedef UpwindInterpolant<XVolField,XSurfZField> UpwindXVolXSurfZ;
    opDB.register_new_operator<UpwindXVolXSurfZ>(scinew UpwindXVolXSurfZ(dim, bcPlus ));

    typedef UpwindInterpolant<YVolField,YSurfXField> UpwindYVolYSurfX;
    opDB.register_new_operator<UpwindYVolYSurfX>(scinew UpwindYVolYSurfX(dim, bcPlus ));
    
    typedef UpwindInterpolant<YVolField,YSurfYField> UpwindYVolYSurfY;
    opDB.register_new_operator<UpwindYVolYSurfY>(scinew UpwindYVolYSurfY(dim, bcPlus ));
    
    typedef UpwindInterpolant<YVolField,YSurfZField> UpwindYVolYSurfZ;
    opDB.register_new_operator<UpwindYVolYSurfZ>(scinew UpwindYVolYSurfZ(dim, bcPlus ));
    
    typedef UpwindInterpolant<ZVolField,ZSurfXField> UpwindZVolZSurfX;
    opDB.register_new_operator<UpwindZVolZSurfX>(scinew UpwindZVolZSurfX(dim, bcPlus ));
    
    typedef UpwindInterpolant<ZVolField,ZSurfYField> UpwindZVolZSurfY;
    opDB.register_new_operator<UpwindZVolZSurfY>(scinew UpwindZVolZSurfY(dim, bcPlus ));
    
    typedef UpwindInterpolant<ZVolField,ZSurfZField> UpwindZVolZSurfZ;
    opDB.register_new_operator<UpwindZVolZSurfZ>(scinew UpwindZVolZSurfZ(dim, bcPlus ));    

    //--------------------------------------------------------
    // FLUX LIMITER interpolants - phi volume to phi surface
    //--------------------------------------------------------    
    typedef FluxLimiterInterpolant<SVolField,SSurfXField> FluxLimiterSVolSSurfX;
    opDB.register_new_operator<FluxLimiterSVolSSurfX>(scinew FluxLimiterSVolSSurfX(dim, bcPlus) );
    
    typedef FluxLimiterInterpolant<SVolField,SSurfYField> FluxLimiterSVolSSurfY;
    opDB.register_new_operator<FluxLimiterSVolSSurfY>(scinew FluxLimiterSVolSSurfY(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<SVolField,SSurfZField> FluxLimiterSVolSSurfZ;
    opDB.register_new_operator<FluxLimiterSVolSSurfZ>(scinew FluxLimiterSVolSSurfZ(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<XVolField,XSurfXField> FluxLimiterXVolXSurfX;
    opDB.register_new_operator<FluxLimiterXVolXSurfX>(scinew FluxLimiterXVolXSurfX(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<XVolField,XSurfYField> FluxLimiterXVolXSurfY;
    opDB.register_new_operator<FluxLimiterXVolXSurfY>(scinew FluxLimiterXVolXSurfY(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<XVolField,XSurfZField> FluxLimiterXVolXSurfZ;
    opDB.register_new_operator<FluxLimiterXVolXSurfZ>(scinew FluxLimiterXVolXSurfZ(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<YVolField,YSurfXField> FluxLimiterYVolYSurfX;
    opDB.register_new_operator<FluxLimiterYVolYSurfX>(scinew FluxLimiterYVolYSurfX(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<YVolField,YSurfYField> FluxLimiterYVolYSurfY;
    opDB.register_new_operator<FluxLimiterYVolYSurfY>(scinew FluxLimiterYVolYSurfY(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<YVolField,YSurfZField> FluxLimiterYVolYSurfZ;
    opDB.register_new_operator<FluxLimiterYVolYSurfZ>(scinew FluxLimiterYVolYSurfZ(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<ZVolField,ZSurfXField> FluxLimiterZVolZSurfX;
    opDB.register_new_operator<FluxLimiterZVolZSurfX>(scinew FluxLimiterZVolZSurfX(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<ZVolField,ZSurfYField> FluxLimiterZVolZSurfY;
    opDB.register_new_operator<FluxLimiterZVolZSurfY>(scinew FluxLimiterZVolZSurfY(dim, bcPlus ));
    
    typedef FluxLimiterInterpolant<ZVolField,ZSurfZField> FluxLimiterZVolZSurfZ;
    opDB.register_new_operator<FluxLimiterZVolZSurfZ>(scinew FluxLimiterZVolZSurfZ(dim, bcPlus ));    
    
    //--------------------------------------------------------
    // scalar surface to staggered volumes (pressure gradients)
    //--------------------------------------------------------
    opDB.register_new_operator<InterpSSurfXXVol>( InterpSSurfXXVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSSurfYYVol>( InterpSSurfYYVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpSSurfZZVol>( InterpSSurfZZVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    
    //--------------------------------------------------------
    // staggered volume to scalar surface
    //--------------------------------------------------------
    typedef OperatorTypeBuilder< Interpolant, XVolField, SVolField >::type  InterpXVolSVol;
    typedef OperatorTypeBuilder< Interpolant, YVolField, SVolField >::type  InterpYVolSVol;
    typedef OperatorTypeBuilder< Interpolant, ZVolField, SVolField >::type  InterpZVolSVol;    
    opDB.register_new_operator<InterpXVolSVol>( InterpXVolSVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpYVolSVol>( InterpYVolSVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    opDB.register_new_operator<InterpZVolSVol>( InterpZVolSVol::Assembler( dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    

    //--------------------------------------------------------
    // scratch operators
    //--------------------------------------------------------
//     opDB.register_new_operator<ScratchSVol>( ScratchSVol::Assembler( dim, XDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSVol>( ScratchSVol::Assembler( dim, YDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSVol>( ScratchSVol::Assembler( dim, ZDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                          
//     opDB.register_new_operator<ScratchXVol>( ScratchXVol::Assembler( dim, XDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchXVol>( ScratchXVol::Assembler( dim, YDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchXVol>( ScratchXVol::Assembler( dim, ZDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                          
//     opDB.register_new_operator<ScratchYVol>( ScratchYVol::Assembler( dim, XDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchYVol>( ScratchYVol::Assembler( dim, YDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchYVol>( ScratchYVol::Assembler( dim, ZDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                          
//     opDB.register_new_operator<ScratchZVol>( ScratchZVol::Assembler( dim, XDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchZVol>( ScratchZVol::Assembler( dim, YDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchZVol>( ScratchZVol::Assembler( dim, ZDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
    

//     // scratch grad ops - need to test these.
 
//     opDB.register_new_operator<ScratchSVolSSurfX>( ScratchSVolSSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSVolSSurfY>( ScratchSVolSSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSVolSSurfZ>( ScratchSVolSSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                                       
//     opDB.register_new_operator<ScratchXVolXSurfX>( ScratchXVolXSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchXVolXSurfY>( ScratchXVolXSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchXVolXSurfZ>( ScratchXVolXSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                                       
//     opDB.register_new_operator<ScratchYVolYSurfX>( ScratchYVolYSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchYVolYSurfY>( ScratchYVolYSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchYVolYSurfZ>( ScratchYVolYSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
                                                                                                                                       
//     opDB.register_new_operator<ScratchZVolZSurfX>( ScratchZVolZSurfX::Assembler( spacing[0], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchZVolZSurfY>( ScratchZVolZSurfY::Assembler( spacing[1], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchZVolZSurfZ>( ScratchZVolZSurfZ::Assembler( spacing[2], dim, bcPlus[0], bcPlus[1], bcPlus[2] ) );

//     // bc operators
//     opDB.register_new_operator<ScratchSSurfX>( ScratchSSurfX::Assembler( dim, XDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSSurfY>( ScratchSSurfY::Assembler( dim, YDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
//     opDB.register_new_operator<ScratchSSurfZ>( ScratchSSurfZ::Assembler( dim, ZDIR::value, bcPlus[0], bcPlus[1], bcPlus[2] ) );
  }

} // namespace Wasatch
