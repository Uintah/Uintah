/*
 * The MIT License
 *
 * Copyright (c) 2013 The University of Utah
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

//-- C++ Includes --//
#include <map>
#include <set>
#include <list>
#include <string>
#include <iostream>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FVStaggeredBCTools.h>

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/ComputeSet.h> // used for Uintah::PatchSet

//-- Wasatch Includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"
#include "Operators/OperatorTypes.h"

//-- Debug Stream --//
#include <Core/Util/DebugStream.h>

static SCIRun::DebugStream dbgbc("WASATCH_BC", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBC  if( DBC_BC_ON  ) dbgbc

namespace Wasatch {
  
  typedef std::map<std::string, std::set<std::string> > BCFunctorMap;

//  enum AtomicBCTypeEnum
//  {
//    DIRICHLET,
//    NEUMANN,
//    UNSUPPORTED
//  }
  
  //****************************************************************************
  /**
   *  @enum   BoundaryTypeEnum
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Enum that specifies the types of boundary conditions supported in Wasatch.
   While it all boils down to setting a Dirichlet or Neumnann condition on the boundary,
   the remaining enums (WALL, etc...) provide a logical mechanism to set other Dirichlet 
   and Neumann conditions on auxiliary variables.
   */
  //****************************************************************************  
  enum BoundaryTypeEnum
  {
    DIRICHLET,
    NEUMANN,
    WALL,
    VELOCITYINLET,
    PRESSURE,
    OUTFLOW,
    UNSUPPORTED
  };

  //****************************************************************************
  /**
   *  @enum   BCValueTypeEnum
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Specifies the boundary condition value datatypes supported by Wasatch.
   */
  //****************************************************************************  
  enum BCValueTypeEnum
  {
    DOUBLE_TYPE,
    VECTOR_TYPE,
    FUNCTOR_TYPE,
    INVALID_TYPE
  };

  //****************************************************************************
  /**
   *  @struct BoundarySpec
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores boundary condition information in a convenient way that Wasatch can manipulate.
   */
  //****************************************************************************
  struct BoundarySpec
  {
    Uintah::Patch::FaceType face; // x-minus, x-plus, y-minus, y-plus, z-minus, z-plus
    int                     patchID;
    std::string             bcName;      // name of the boundary on which this BC is applied
    std::string             varName;     // mame of the variable on which we want to apply a BC
    std::string             functorName; // name of the functor applied as bc
    double                  value;            // boundary value for this variable
    SpatialOps::structured::Numeric3Vec<double> vecValue;
    BoundaryTypeEnum        bcType; // boundary type for this variable
    BCValueTypeEnum         bcValType;
  };

  //****************************************************************************
  /**
   *  @struct BoundaryIterators
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores the domain boundary iterators necessary for setting boundary conditions.
   */
  //****************************************************************************
  struct BoundaryIterators
  {
    std::vector<SpatialOps::structured::IntVec> extraBndCells;     // iterator for extra cells
    std::vector<SpatialOps::structured::IntVec> extraPlusBndCells; // iterator for extra cells
    std::vector<SpatialOps::structured::IntVec> interiorBndCells;  // iterator for interior cells
  };
  
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
    
    typedef typename Ops::GradX   NeumannX;
    typedef typename Ops::GradY   NeumannY;
    typedef typename Ops::GradZ   NeumannZ;
  };
  
  //
  template< typename FieldT>
  struct BCOpTypeSelector : public BCOpTypeSelectorBase<FieldT>
  { };
  
  // partial specialization with inheritance for XVolFields
  template<>
  struct BCOpTypeSelector<XVolField> : public BCOpTypeSelectorBase<XVolField>
  {
    typedef SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientX, XVolField, XVolField >::type NeumannX;
  };
  
  // partial specialization with inheritance for YVolFields
  template<>
  struct BCOpTypeSelector<YVolField> : public BCOpTypeSelectorBase<YVolField>
  {
    typedef SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientY, YVolField, YVolField >::type NeumannY;
  };
  
  // partial specialization with inheritance for ZVolFields
  template<>
  struct BCOpTypeSelector<ZVolField> : public BCOpTypeSelectorBase<ZVolField>
  {
    typedef SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientZ, ZVolField, ZVolField >::type NeumannZ;
  };
  
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<XVolField>::XFace>
  {
  public:
    typedef SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::XSurfXField, SpatialOps::structured::XVolField >::type DirichletX;
    typedef SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::XSurfXField, SpatialOps::structured::XVolField >::type NeumannX;
  };
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<YVolField>::YFace>
  {
    typedef SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::YSurfYField, SpatialOps::structured::YVolField >::type DirichletY;
    typedef SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::YSurfYField, SpatialOps::structured::YVolField >::type NeumannY;
  };
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<ZVolField>::ZFace>
  {
    typedef SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::ZSurfZField, SpatialOps::structured::ZVolField >::type DirichletZ;
    typedef SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::ZSurfZField, SpatialOps::structured::ZVolField >::type NeumannZ;
  };

  /**
   *  \class   BCHelper
   *  \author  Tony Saad
   *  \date    September, 2013
   *
   *  The BCHelper class provides a centralized approach to dealing with boundary
   *  conditions. The model adopted for our boundary condition implementation
   *  relies on the basic assumption that all boundary specification within a
   *  <Face> specification in a ups input file belong to the same boundary.
   *  This is the essential assumption on which this entire class is built.
   *
   *  The class operates in the following manner. After Uintah performs its
   *  input-file setup, it automatically constructs BC-related objects and
   *  iterators based on input specification. The Uintah model operates as
   *  follows:
   *   <ul>
   *   <li> Every Uintah::Patch is a logical box that may contain boundary faces.
   *   <li> For every face, Uintah creates a Uintah::BCDataArray object.
   *   <li> The BCDataArray class stores ALL the information related to the boundary
   *    condition(s) specified at that boundary face.
   *   </ul>
   *  For example, if a boundary face consists of two geometric objects (side,
   *  and circle), then the BCDataArray contains information about the side and
   *  the circle, their iterators, the boundary conditions specified by the user
   *  on each of these boundaries (i.e. side and circle). Uintah refers to these
   *  sub-boundaries as "children". Then, to summarize, for each boundary face,
   *  Uintah constructs a BCDataArray object. The BCDataArray object contains
   *  information about the children specified at a face.
   *
   *  Now each child is stored as a BCGeom object. Each BCGeom object contains
   *  an iterator that lists the extra cells at that geom object as well as a
   *  Uintah::BCData object. The Uintah::BCData class contains information about
   *  the boundaries specified by the user.
   */

  class BCHelper {

  private:
    typedef SpatialOps::structured::IntVec                     IntVecT;            // SpatialOps IntVec
    typedef std::map <int, BoundaryIterators                 > patchIDBndItrMapT; // temporary map that stores boundary iterators per patch id
    typedef std::map <std::string, patchIDBndItrMapT         > MaskMapT;
    typedef std::map <std::string, std::vector<BoundarySpec> > BCMapT;
    
    const Uintah::PatchSet*    const localPatches_;
    const Uintah::MaterialSet* const materials_;
    const PatchInfoMap&        patchInfoMap_;
    const BCFunctorMap&        bcFunctorMap_;
    GraphCategories&           grafCat_;
    Category                   taskCat_;
    
    // This map stores the iterators associated with each boundary condition name.
    // The iterators are stored in a map keyed by patch ID. a single iterator will be associated
    // with each boundary (aka child)
    MaskMapT                   bcNamePatchIDMaskMap_;
    // Mapt that stores the BoundarySpecification for a given variable. the map is indexed by
    // variable name! a collection of boundary specification will be associated with each variable
    BCMapT                     varNameBoundarySpecMap_;
    
    template<typename FieldT>
    const std::vector<IntVecT>* get_extra_bnd_mask( const BoundarySpec& myBCSpec ) const;

    template<typename FieldT>
    const std::vector<IntVecT>* get_interior_bnd_mask( const BoundarySpec& myBCSpec ) const;

    void add_boundary_iterator( const BoundaryIterators& myIters,
                                const std::string& bcName,
                                const int& patchID );
    
    void add_boundary_condition( const BoundarySpec& bcSpec );
    
    void parse_boundary_conditions();
    
    // apply a boundary condition on a field given another one. here, the srcVarName designates the
    // variable from which we should infer other data
    void add_auxiliary_boundary_condition( const std::string& srcVarName,
                                           const BoundarySpec& targetBCSpec );
    
  public:
    
    BCHelper( const Uintah::PatchSet* const localPatches,
              const Uintah::MaterialSet* const materials,
              const PatchInfoMap& patchInfoMap,
              GraphCategories& grafCat,
              const BCFunctorMap& bcFunctorMap );
    
    ~BCHelper();

    /**
     *  \brief Specifies the graph/task category from which the BCHelper will extract expressions
     *  and set boundary conditions on. This function is really useful when setting a series of
     *  boundary conditions, back-to-back, as it allows one to use the simple apply_boundary_condition
     *  interface without having to specify the graph category every time.
     *
     *  \param taskCat A Category enum designating the graph category.
     *
     */
    void set_task_category( const Category taskCat );
    
    /**
     *  \brief Function that allows one to add an auxiliary boundary condition based on an existing
     *  one. This situation typically arises when one needs to specify an auxiliary boundary condition
     *  to complement one specified by the user. For example, at a moving wall, the user typically specifies
     *  velocity but boundary conditions must be specified on the momentum RHS, among other things.
     *
     *  \param srcVarName A string designating the name of the variable on which one desires
     *  to "template" the auxiliary boundary. By "template" we mean that the new boundary condition will
     *  be set on the same boundary(ies), patches, etc...
     *
     *  \param newVarName An std::string designating the name of the auxiliary variable, i.e. the
     *  variable for which we want to create a new boundary condition.
     *
     *  \param newValue The value (double) of the auxiliary variable at the boundary.
     *
     *  \param newBCType The type (DIRICHLET/NEUMANN) of the auxiliary bc.
     */    
    void add_auxiliary_boundary_condition( const std::string& srcVarName,
                                           const std::string& newVarName,
                                           const double& newValue,
                                           const BoundaryTypeEnum newBCType );
    /**
     *  \brief Function that allows one to add an auxiliary boundary condition based on an existing
     *  one. This situation typically arises when one needs to specify an auxiliary boundary condition
     *  to complement one specified by the user. For example, at a moving wall, the user typically specifies
     *  velocity but boundary conditions must be specified on the momentum RHS, among other things.
     *
     *  \param srcVarName A string designating the name of the variable on which one desires
     *  to "template" the auxiliary boundary. By "template" we mean that the new boundary condition will
     *  be set on the same boundary(ies), patches, etc...
     *
     *  \param newVarName An std::string designating the name of the auxiliary variable, i.e. the
     *  variable for which we want to create a new boundary condition.
     *
     *  \param functorName The name of the functor that applies to the auxiliary boundary.
     *
     *  \param newBCType The type (DIRICHLET/NEUMANN) of the auxiliary bc.
     */
    void add_auxiliary_boundary_condition( const std::string& srcVarName,
                                           const std::string& newVarName,
                                           const std::string& functorName,
                                           const BoundaryTypeEnum newBCType );

    
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
                                   bool setOnExtraOnly=false );

    /**
     *  \brief Lightweight version of the the previous function to apply boundary conditions on a
     *  given expression. This function does NOT take CATEGORY enum that specifies the graph in which
     *  the BC is to be applied. The caveat here is that you MUST specify the default taskCategory
     *  by using BCHelper::set_task_category
     *
     *  \param varTag The Expr::Tag of the expression on which the boundary
     *  condition is to be applied.
     *
     *
     *  \param setOnExtraOnly Optional boolean flag - specifies whether to set the boundary value
     *  DIRECTLY on the extra cells without doing averaging using interior cells. This is only useful
     *  for DIRICHLET boundary conditions.
     *
     *  \warning BE VERY CAREFUL WHEN USING THIS! YOU MUST SET THE TASK CATEGORY BEFORE, OTHERWISE,
     *  YOU WILL RISK USING AN INVALID TASK CATEGORY!!!
     *
     */
    template<typename FieldT>
    void apply_boundary_condition( const Expr::Tag& varTag,
                                   bool setOnExtraOnly=false );  
    /**
     *  \brief Print boundary conditions summary.
     *
     */
    void print() const;
    
  }; // class BCHelper

} // namespace Wasatch

#endif /* defined(WASATCH_BC_HELPER) */
