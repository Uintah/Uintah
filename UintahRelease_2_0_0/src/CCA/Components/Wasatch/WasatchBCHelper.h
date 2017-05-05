/*
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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
/* ----------------------------------------------------------------------------
 ########   ######  ##     ## ######## ##       ########  ######## ########
 ##     ## ##    ## ##     ## ##       ##       ##     ## ##       ##     ##
 ##     ## ##       ##     ## ##       ##       ##     ## ##       ##     ##
 ########  ##       ######### ######   ##       ########  ######   ########
 ##     ## ##       ##     ## ##       ##       ##        ##       ##   ##
 ##     ## ##    ## ##     ## ##       ##       ##        ##       ##    ##
 ########   ######  ##     ## ######## ######## ##        ######## ##     ##
 ----------------------------------------------------------------------------*/

#ifndef WASATCH_BC_HELPER
#define WASATCH_BC_HELPER

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>

// -- NSCBC Includes -- //
#include <nscbc/CharacteristicBCBuilder.h>
#include <nscbc/TagManager.h>

/**
 * \file WasatchBCHelper.h
 */

namespace WasatchCore {
  
  //****************************************************************************
  /**
   *  @struct BCOpTypeSelectorBase
   *  @author Tony Saad
   *
   *  @brief This templated struct is used to simplify boundary
   *         condition operator selection.
   */
  //****************************************************************************
  template< typename FieldT>
  struct BCOpTypeSelectorBase
  {
  private:
    typedef OpTypes<FieldT> Ops;
    
  public:
    typedef typename Ops::InterpC2FX   DirichletX;
    typedef typename Ops::InterpC2FY   DirichletY;
    typedef typename Ops::InterpC2FZ   DirichletZ;

    typedef typename Ops::InterpC2FX   InterpX;
    typedef typename Ops::InterpC2FY   InterpY;
    typedef typename Ops::InterpC2FZ   InterpZ;
    
    typedef typename Ops::GradX   NeumannX;
    typedef typename Ops::GradY   NeumannY;
    typedef typename Ops::GradZ   NeumannZ;
  };
  
  //
  template< typename FieldT>
  struct BCOpTypeSelector : public BCOpTypeSelectorBase<FieldT>
  { };

  // partial specialization for particles. Use SVolField to get this to compile. Classic Boundary operators are meaningless for particles.
  template<>
  struct BCOpTypeSelector<ParticleField> : public BCOpTypeSelectorBase<SVolField>
  { };

  // partial specialization with inheritance for XVolFields
  template<>
  struct BCOpTypeSelector<XVolField> : public BCOpTypeSelectorBase<XVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, XVolField, XVolField >::type InterpX;
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientX, XVolField, XVolField >::type NeumannX;
  };
  
  // partial specialization with inheritance for YVolFields
  template<>
  struct BCOpTypeSelector<YVolField> : public BCOpTypeSelectorBase<YVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, YVolField, YVolField >::type InterpY;
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientY, YVolField, YVolField >::type NeumannY;
  };
  
  // partial specialization with inheritance for ZVolFields
  template<>
  struct BCOpTypeSelector<ZVolField> : public BCOpTypeSelectorBase<ZVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, ZVolField, ZVolField >::type InterpZ;
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientZ, ZVolField, ZVolField >::type NeumannZ;
  };
  
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<SVolField>::XFace>
  {
    typedef SpatialOps::SSurfXField FieldT;
  public:
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletX;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpX;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannX;
    
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletY;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpY;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannY;

    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletZ;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpZ;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannZ;

  };

  template<>
  struct BCOpTypeSelector<FaceTypes<SVolField>::YFace>
  {
    typedef SpatialOps::SSurfYField FieldT;
  public:
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletX;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpX;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannX;
    
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletY;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpY;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannY;
    
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletZ;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpZ;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannZ;
  };
  
  template<>
  struct BCOpTypeSelector<FaceTypes<SVolField>::ZFace>
  {
    typedef SpatialOps::SSurfZField FieldT;
  public:
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletX;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpX;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannX;
    
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletY;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpY;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannY;
    
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type DirichletZ;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, FieldT, SpatialOps::SVolField >::type InterpZ;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  FieldT, SpatialOps::SVolField >::type NeumannZ;
  };


  //
  template<>
  struct BCOpTypeSelector<FaceTypes<XVolField>::XFace>
  {
  public:
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::XSurfXField, SpatialOps::XVolField >::type DirichletX;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::XSurfXField, SpatialOps::XVolField >::type InterpX;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  SpatialOps::XSurfXField, SpatialOps::XVolField >::type NeumannX;
  };
  
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<YVolField>::YFace>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::YSurfYField, SpatialOps::YVolField >::type DirichletY;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::YSurfYField, SpatialOps::YVolField >::type InterpY;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  SpatialOps::YSurfYField, SpatialOps::YVolField >::type NeumannY;
  };
  
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<ZVolField>::ZFace>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::ZSurfZField, SpatialOps::ZVolField >::type DirichletZ;
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::ZSurfZField, SpatialOps::ZVolField >::type InterpZ;
    typedef SpatialOps::OperatorTypeBuilder<Divergence,  SpatialOps::ZSurfZField, SpatialOps::ZVolField >::type NeumannZ;
  };

  
  //****************************************************************************
  /**
   *  @struct NSCBCSpec
   *  @author Tony Saad
   *  @date   Nov 2016
   *
   *  @brief  Stores information about NSCBC spec (far field etc...)
   */
  //****************************************************************************
  struct NSCBCSpec
  {
    bool enableNSCBC;
    double pFar; // far field relaxation pressure
    double lx;   // domain length in the x direction
    double ly;   // domain length in the y direction
    double lz;   // domain length in the z direction
  };

  //****************************************************************************
  /**
   *  \class   WasatchBCHelper
   *  \author  Tony Saad
   *  \date    September, 2015
   *  \brief   The WasatchBCHelper class derives from the BCHelper and allows the application of boundary
   conditions in Wasatch only.
   *
   */
  //****************************************************************************
  class WasatchBCHelper : public BCHelper {
    
  private:
    const PatchInfoMap&        patchInfoMap_      ;
    BCFunctorMap&              bcFunctorMap_      ;
    GraphCategories&           grafCat_           ;
    
    typedef std::map <int, NSCBC::BCBuilder<SVolField>* > PatchIDNSCBCBuilderMapT;  // temporary typedef map that stores boundary iterators per patch id: Patch ID -> Bnd Iterators
    typedef std::map <std::string, PatchIDNSCBCBuilderMapT    > NSCBCMapT         ;  // boundary name -> (patch ID -> NSCBC Builder )
    
    NSCBCMapT nscbcBuildersMap_;
    
    NSCBCSpec nscbcSpec_;
    
  public:
    
    WasatchBCHelper( const Uintah::LevelP& level,
                    Uintah::SchedulerP& sched,
                    const Uintah::MaterialSet* const materials,
                    const PatchInfoMap& patchInfoMap,
                    GraphCategories& grafCat,
                    BCFunctorMap& bcFunctorMap,
                    Uintah::ProblemSpecP wasatchSpec);
        
    ~WasatchBCHelper();

    // The necessary evils of the pressure expression.
    /**
     *  \brief This passes the BCHelper to ALL pressure expressions in the factory (per patch that is)
     */
    void synchronize_pressure_expression();

    /**
     *  \brief This function updates the pressure coefficient matrix for boundary conditions
     */
    void update_pressure_matrix( Uintah::CCVariable<Uintah::Stencil7>& pMatrix,
                                 const SVolField* const svolFrac,
                                 const Uintah::Patch* patch );

    /**
     *  \brief This function applies the boundary conditions on the pressure. The pressure is a special
     * expression in that modifiers can't work with it directly since we have to schedule the hypre
     * solver first and then apply the BCs
     */
    void apply_pressure_bc( SVolField& pressureField,
                            const Uintah::Patch* patch );

    /**
     *  \ingroup WasatchCore
     *
     *  \brief Function that updates poisson rhs when boundaries are present.
     *
     *  \param pressureRHS A reference to the poisson RHS field. This should be
     *  a MODIFIABLE field since it will be updated using bcs on the poisson field.
     *
     *  \param patch A pointer to the current patch. If the patch does NOT contain
     *  the reference cells, then nothing is set.
     */
    void update_pressure_rhs( SVolField& pressureRHS,
                              const Uintah::Patch* patch );


    /**
     *  \brief Key member function that applies a boundary condition on a given expression.
     *
     *  \param varTag The Expr::Tag of the expression on which the boundary
     *   condition is to be applied.
     *
     *  \param taskCat Specifies on which graph to apply this boundary condition.
     *
     *  \param setOnExtraOnly Optional boolean flag - specifies whether to set the boundary value
     *  DIRECTLY on the extra cells without doing averaging using interior cells. This is only useful
     *  for DIRICHLET boundary conditions.
     */
    template<typename FieldT>
    void apply_boundary_condition( const Expr::Tag& varTag,
                                   const Category& taskCat,
                                   const bool setOnExtraOnly=false );
    
    /**
     *  \brief Key member function that applies a boundary condition on a given expression.
     *
     *  \param varTag The Expr::Tag of the expression on which the boundary
     *   condition is to be applied.
     *
     *  \param taskCat Specifies on which graph to apply this boundary condition.
     *
     *  \param setOnExtraOnly Optional boolean flag - specifies whether to set the boundary value
     *  DIRECTLY on the extra cells without doing averaging using interior cells. This is only useful
     *  for DIRICHLET boundary conditions.
     */
    void apply_nscbc_boundary_condition( const Expr::Tag& varTag,
                                         const NSCBC::TransportVal& quantity,
                                         const Category& taskCat  );

    /**
     *  \brief Allows one to inject dummy dependencies to help with boundary condition expressions.
     *  \param targetTag    The Expression tag on which we want to attach a new dependency
     *  \param dependencies A TagList of new dependencies to attach to targetTag
     *  \param taskCat      The task in which the dependencies are to be added
     *
     */
    template<typename SrcT, typename TargetT >
    void create_dummy_dependency(const Expr::Tag& targetTag,
                                 const Expr::TagList dependencies,
                                 const Category taskCat);
    
    template <typename MomDirT>
    void setup_nscbc(const BndSpec& myBndSpec, NSCBC::TagManager nscbcTagMgr, const int jobid);
    
    bool do_nscbc() {return nscbcSpec_.enableNSCBC;}
  }; // class WasatchBCHelper
  
} // namespace WasatchCore

#endif /* defined(WASATCH_BC_HELPER) */
