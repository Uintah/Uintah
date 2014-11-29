/*
 * The MIT License
 *
 * Copyright (c) 2013-2014 The University of Utah
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

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/ComputeSet.h> // used for Uintah::PatchSet
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
//-- Wasatch Includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"
#include "Operators/OperatorTypes.h"

//-- Debug Stream --//
#include <Core/Util/DebugStream.h>

static SCIRun::DebugStream dbgbc("WASATCH_BC", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBC  if( DBC_BC_ON  ) dbgbc

/**
 * \file BCHelper.h
 */

namespace Wasatch {
  
  // !!! ACHTUNG !!!
  // !!! READ THE NOMENCLATURE ASSUMPTIONS BEFORE PROCEEDING WITH THIS CLASS !!!
  /* 
     Hi. This class is based on the following assumptions:
   1. We distinguish between boundaries and boundary conditions. 
   2. We associate boundaries with physical domain contraints. 
   3. Boundaries include Walls, Inlets, Velocity inlets, mass flow inlets, outflows, pressure outlets... or
     any physically relevant constraint on the flow-field. Boundaries do NOT include DIRICHLET, NEUMANN,
     or ROBIN conditions.
   4. We associate boundary conditions with the mathematical specification or mathematical constraints
     specified on variables at boundaries. 
   5. Boundary conditions include DIRICHLET, NEUMANN, and ROBIN
     conditions. 
   */
  // Nomenclature: Boundary/Bnd/Bound designates a PHYSICAL BOUNDARY
  //               Boundary Condition/BndCond/BC designates a BOUNDARY CONDITION

  typedef std::map<std::string, std::set<std::string> > BCFunctorMap;

  //****************************************************************************
  /**
   *  @enum   BndCondTypeEnum
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Enum that specifies the types of boundary-conditions supported in Wasatch.
   We support Dirichlet and Neumnann conditions on the boundary.
   */
  //****************************************************************************
  enum BndCondTypeEnum
  {
    DIRICHLET,
    NEUMANN,
    UNSUPPORTED
  };
  
  BndCondTypeEnum   select_bc_type_enum( const std::string& bcTypeStr );
  std::string bc_type_enum_to_string( const BndCondTypeEnum bcTypeEnum );

  template<typename OST>
  OST& operator<<( OST& os, const BndCondTypeEnum bcTypeEnum );

  
  //****************************************************************************
  /**
   *  @enum   BndTypeEnum
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Enum that specifies the types of boundaries supported in Wasatch.
   *  Boundaries represent physical domain boundaries and can be of type Wall, Inlet, etc...
   *  They can be thought of as physical, user-friendly boundaries types. These types, specified
   *  in the input file, will be used to make logical decisions on the sanity of boundary conditions
   *  specified by the user. They are also used to infer auxiliary boundary conditions.
   *
   *  The boundary type is specified by the user through the input file, for example:
   *  \verbatim<Face side="x+" name="outlet" type="Outflow"/>\verbatim
   *  All types specified in the input file are Capitalized (first letter only).
   *  If the user doesn't specify a type, then Wasatch will assume that the boundary type is USER, i.e.
   *  the user specifies bcs on any quantity as long as Wasatch applies a bc on that quantity.
   */
  //****************************************************************************
  enum BndTypeEnum
  {
    WALL,     ///< Stationary wall BC. Zero velocity (and momentum).
    VELOCITY, ///< Velocity specification: can be used for inlets or moving walls.
    OPEN,     ///< OPEN boundary condition. a bit complicated to explain but namely mimics a boundary open to the atmosphere.
    OUTFLOW,  ///< OUTFLOW boundary condition. encourages the flow to exit and reduces reflections.
    USER,     ///< User specified bc. The user can specify BCs on any quantity they desire, as long as Wasatch calls apply_boundary_condition on that quantity.
    INVALID
  };
  
  BndTypeEnum       select_bnd_type_enum( const std::string& bcTypeStr );
  const std::string bnd_type_enum_to_string( const BndTypeEnum bcTypeEnum );

  template<typename OST>
  OST& operator<<( OST& os, const BndTypeEnum bcTypeEnum );
  
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
    FUNCTOR_TYPE, // string
    INVALID_TYPE
  };
  
  //****************************************************************************
  /**
   *  @struct BndCondSpec
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores boundary condition information in a convenient way that Wasatch can manipulate.
   */
  //****************************************************************************  
  struct BndCondSpec
  {
    std::string      varName;     // mame of the variable on which we want to apply a BC
    std::string      functorName; // name of the functor applied as bc
    double           value;       // boundary value for this variable
    BndCondTypeEnum  bcType;      // bc type: DIRICHLET, NEUMANN
    BCValueTypeEnum  bcValType;   // value type: DOUBLE, FUNCTOR    
    
    // compare based on ALL the members of this struct
    bool operator==(const BndCondSpec& l) const
    {
      return (   l.varName == varName
              && l.functorName == functorName
              && l.value == value
              && l.bcType == bcType
              && l.bcValType == bcValType);
    };

    // compare based on the varname only
    bool operator==(const std::string& varNameNew) const
    {
      return ( varNameNew == varName);
    };

    // print
    void print() const
    {
      using namespace std;
      cout << "  var:   " << varName << endl
           << "  type:  " << bcType << endl
           << "  value: " << value << endl;
    };

    // check if the user is applying a functor in this boundary condition
    bool is_functor() const
    {
      return (bcValType == FUNCTOR_TYPE);
    };
  };

  //****************************************************************************
  /**
   *  @struct BndSpec
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores boundary information in a convenient way that Wasatch can manipulate.
   */
  //****************************************************************************
  struct BndSpec
  {
    std::string              name;      // name of the boundary
    Uintah::Patch::FaceType  face;      // x-minus, x-plus, y-minus, y-plus, z-minus, z-plus
    BndTypeEnum              type;      // Wall, inlet, etc...
    std::vector<int>         patchIDs;  // List of patch IDs that this boundary lives on.
                                        //Note that a boundary is typically split between several patches.
    Uintah::BCGeomBase::ParticleBndSpec particleBndSpec;    
    std::vector<BndCondSpec> bcSpecVec; // List of ALL the BCs applied at this boundary

    // returns true if this Boundary has parts of it on patchID
    bool has_patch(const int& patchID) const
    {
      return std::find(patchIDs.begin(), patchIDs.end(), patchID) != patchIDs.end();
    };
    
    // find the BCSpec associated with a given variable name
    const BndCondSpec* find(const std::string& varName) const
    {
      std::vector<BndCondSpec>::const_iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
      if (it != bcSpecVec.end()) {
        return &(*it);
      } else {
        return NULL;
      }
    };
    
    // find the BCSpec associated with a given variable name - non-const version
    const BndCondSpec* find(const std::string& varName)
    {
      std::vector<BndCondSpec>::iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
      if (it != bcSpecVec.end()) {
        return &(*it);
      } else {
        return NULL;
      }
    };
    
    // check whether this boundary has any bcs specified for varName
    bool has_field(const std::string& varName) const
    {
      std::vector<BndCondSpec>::const_iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
      if (it != bcSpecVec.end()) {
        return true;
      } else {
        return false;
      }
    };
    
    // print information about this boundary
    void print() const
    {
      using namespace std;
      cout << "Boundary: " << name << " face: " << face << " BndType: " << type << endl;
      for (vector<BndCondSpec>::const_iterator it=bcSpecVec.begin(); it != bcSpecVec.end(); ++it) {
        (*it).print();
      }
    };
  };
  
  typedef std::map <std::string, BndSpec> BndMapT;
  
  //****************************************************************************
  /**
   *  @struct BoundaryIterators
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores the domain boundary iterators necessary for setting boundary conditions.
   *  For particles, we store a pointer to a list of particles near a given boundary. That list
   is updated by the ParticlesHelper object. This external list stores the particle index that is
   near the boundary, NOT the particle ID.
   */
  //****************************************************************************
  struct BoundaryIterators
  {
    std::vector<SpatialOps::IntVec> extraBndCells;          // iterator for extra cells. These are zero-based on the extra cell
    std::vector<SpatialOps::IntVec> extraPlusBndCells;      // iterator for extra cells on plus faces (staggered fields). These are zero-based on the extra cell.
    std::vector<SpatialOps::IntVec> interiorBndCells;       // iterator for interior cells. These are zero-based on the extra cell
    
    std::vector<SpatialOps::IntVec> neboExtraBndCells;      // iterator for extra cells. These are zero-based on the first interior cell.
    std::vector<SpatialOps::IntVec> neboExtraPlusBndCells;  // iterator for extra cells on plus faces (staggered fields). These are zero-based on the first interior cell.
    std::vector<SpatialOps::IntVec> neboInteriorBndCells;   // iterator for interior cells. These are zero-based on the first interior cell.

    std::vector<SpatialOps::IntVec> interiorEdgeCells;      // iterator for interior edge (domain edges) cells
    Uintah::Iterator extraBndCellsUintah;                   // We still need the Unitah iterator
    const std::vector<int>* particleIdx;                    // list of particle indices near a given boundary. Given the memory of ALL particles on a given patch, this vector stores
                                                            // the indices of those particles that are near a boundary.
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

  // partial specialization for particles. Use SVolField to get this to compile. Classic Boundary operators are meaningless for particles.
  template<>
  struct BCOpTypeSelector<ParticleField> : public BCOpTypeSelectorBase<SVolField>
  { };

  // partial specialization with inheritance for XVolFields
  template<>
  struct BCOpTypeSelector<XVolField> : public BCOpTypeSelectorBase<XVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientX, XVolField, XVolField >::type NeumannX;
  };
  
  // partial specialization with inheritance for YVolFields
  template<>
  struct BCOpTypeSelector<YVolField> : public BCOpTypeSelectorBase<YVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientY, YVolField, YVolField >::type NeumannY;
  };
  
  // partial specialization with inheritance for ZVolFields
  template<>
  struct BCOpTypeSelector<ZVolField> : public BCOpTypeSelectorBase<ZVolField>
  {
    typedef SpatialOps::OperatorTypeBuilder<SpatialOps::GradientZ, ZVolField, ZVolField >::type NeumannZ;
  };
  
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<XVolField>::XFace>
  {
  public:
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::XSurfXField, SpatialOps::XVolField >::type DirichletX;
    typedef SpatialOps::OperatorTypeBuilder<Divergence, SpatialOps::XSurfXField, SpatialOps::XVolField >::type NeumannX;
  };
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<YVolField>::YFace>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::YSurfYField, SpatialOps::YVolField >::type DirichletY;
    typedef SpatialOps::OperatorTypeBuilder<Divergence, SpatialOps::YSurfYField, SpatialOps::YVolField >::type NeumannY;
  };
  //
  template<>
  struct BCOpTypeSelector<FaceTypes<ZVolField>::ZFace>
  {
    typedef SpatialOps::OperatorTypeBuilder<Interpolant, SpatialOps::ZSurfZField, SpatialOps::ZVolField >::type DirichletZ;
    typedef SpatialOps::OperatorTypeBuilder<Divergence, SpatialOps::ZSurfZField, SpatialOps::ZVolField >::type NeumannZ;
  };

  //****************************************************************************
  /**
   *  \class   BCHelper
   *  \author  Tony Saad
   *  \date    September, 2013
   *
   *  The BCHelper class provides a centralized approach to dealing with boundary
   *  conditions. The model adopted for our boundary condition implementation
   *  relies on the basic assumption that all boundary specification within a
   *  \verbatim<Face>\endverbatim specification in a ups input file belong to
   *  the same boundary. This is the essential assumption on which this entire
   *  class is built.
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
  //****************************************************************************
  class BCHelper {
    
  private:
    typedef SpatialOps::IntVec                IntVecT          ;  // SpatialOps IntVec
    typedef std::map <int, BoundaryIterators            > patchIDBndItrMapT;  // temporary typedef map that stores boundary iterators per patch id: Patch ID -> Bnd Iterators
    typedef std::map <std::string, patchIDBndItrMapT    > MaskMapT         ;  // boundary name -> (patch ID -> Boundary iterators )
    
    const Uintah::PatchSet*    const localPatches_;
    const Uintah::MaterialSet* const materials_   ;
    const PatchInfoMap&        patchInfoMap_      ;
    BCFunctorMap&        bcFunctorMap_      ;
    GraphCategories&           grafCat_           ;
    
    // This map stores the iterators associated with each boundary condition name.
    // The iterators are stored in a map keyed by patch ID. a single iterator will be associated
    // with each boundary (aka child)
    MaskMapT                   bndNamePatchIDMaskMap_;
    
    // bndNameBndSpecMap_ stores BndSpec information for each of the specified boundaries. This
    // map is indexed by the (unique) boundary name.
    BndMapT                    bndNameBndSpecMap_;
    
    /**
     \brief Returns a pointer to the list of particles that are near a given boundary on a given patch ID.
     This pointer is set externally through the ParticlesHelper class.
     */
    const std::vector<int>* get_particles_bnd_mask( const BndSpec& myBndSpec,
                                                    const int& patchID ) const;
    
    template<typename FieldT>
    const std::vector<IntVecT>* get_extra_bnd_mask( const BndSpec& myBndSpec,
                                                    const int& patchID ) const;
    
    template<typename FieldT>
    const std::vector<IntVecT>* get_interior_bnd_mask( const BndSpec& myBndSpec,
                                                      const int& patchID ) const;

    // returns the nebo-friendly extra cell iterator. These are zero-based on the first
    // interior cell. If FieldT is staggered and normal to the boundary, this will
    // return the extra faces (those outside the domain).
    template<typename FieldT>
    const std::vector<IntVecT>* get_nebo_extra_bnd_mask( const BndSpec& myBndSpec,
                                                   const int& patchID ) const;

    // returns the nebo-friendly interior cell iterator. These are zero-based on the first
    // interior cell. If FieldT is staggered and normal to the boundary, this will return
    // the boundary faces.
    template<typename FieldT>
    const std::vector<IntVecT>* get_nebo_interior_bnd_mask( const BndSpec& myBndSpec,
                                                      const int& patchID ) const;

    Uintah::Iterator& get_uintah_extra_bnd_mask( const BndSpec& myBndSpec,
                                                 const int& patchID );
    
    const std::vector<IntVecT>* get_edge_mask( const BndSpec& myBndSpec,
                                                  const int& patchID ) const;
    
    // Add boundary iterator (mask) for boundary "bndName" and patch "patchID"
    void add_boundary_mask( const BoundaryIterators& myIters,
                            const std::string& bndName,
                            const int& patchID );
    
    // Add a new boundary to the list of boundaries specified for this problem. If the boundary
    // already exists, this means that this boundary is shared by several patches. In that case,
    // add the new patchID to the list of patches that this boundary lives on
    void add_boundary( const std::string&      bndName,
                       Uintah::Patch::FaceType face,
                       const BndTypeEnum& bndType,
                       const int               patchID,
                       const Uintah::BCGeomBase::ParticleBndSpec);
    
    // Parse boundary conditions specified through the input file. This function does NOT need
    // an input file since Uintah already parsed and processed most of this information.
    void parse_boundary_conditions();
    
    // apply a boundary condition on a field given another one. here, the srcVarName designates the
    // variable from which we should infer other data
    void add_auxiliary_boundary_condition( const std::string& srcVarName,
                                           BndCondSpec targetBCSpec );
    
  public:
    
    BCHelper( const Uintah::PatchSet* const localPatches,
             const Uintah::MaterialSet* const materials,
             const PatchInfoMap& patchInfoMap,
             GraphCategories& grafCat,
             BCFunctorMap& bcFunctorMap );
        
    ~BCHelper();
    
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
                                          const BndCondTypeEnum newBCType );    
    /**
     *  \brief Adds a boundary condition on a specified boundary
     */
    void add_boundary_condition( const std::string& bndName,
                                 const BndCondSpec& bcSpec   );

    /**
     *  \brief Adds a boundary condition on ALL boundaries
     */
    void add_boundary_condition( const BndCondSpec& bcSpec   );
    

    // The necessary evils of the pressure expression.
    /**
     *  \brief This passes the BCHelper to ALL pressure expressions in the factory (per patch that is)
     */
    void synchronize_pressure_expression();

    /**
     *  \brief This function updates the pressure coefficient matrix for boundary conditions
     */
    void update_pressure_matrix( Uintah::CCVariable<Uintah::Stencil4>& pMatrix,
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
                                  bool setOnExtraOnly=false );
    
    /**
     *  \brief Retrieve a reference to the boundary and boundary condition information stored in this
     *  BCHelper
     */
    BndMapT& get_boundary_information();

    /**
     *  \brief Returns true of the BCHelper on this patch has any physical boundaries
     */
    bool has_boundaries();
    
    /**
     *  \brief Allows one to inject dummy dependencies to help with boundary condition expressions.
     *  \param targetTag    The Expression tag on which we want to attach a new dependency
     *  \param dependencies A TagList of new dependencies to attach to targetTag
     *  \param taskCat      The task in which the dependencies are to be added
     *
     */
    template<typename FieldT>
    void create_dummy_dependency(const Expr::Tag& targetTag,
                                 const Expr::TagList dependencies,
                                 const Category taskCat);
    /**
     *  \brief Print boundary conditions summary.
     *
     */
    void print() const;
    
  }; // class BCHelper
  
} // namespace Wasatch

#endif /* defined(WASATCH_BC_HELPER) */
