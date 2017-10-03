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

#ifndef BC_HELPER
#define BC_HELPER

//-- C++ Includes --//
#include <map>
#include <set>
#include <list>
#include <string>

//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggered.h>

//-- ExprLib Includes --//
#include <expression/ExprLib.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/ComputeSet.h> // used for Uintah::PatchSet
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Debug Stream --//
#include <Core/Util/DebugStream.h>

static Uintah::DebugStream dbgbc("WASATCH_BC", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBC  if( DBC_BC_ON  ) dbgbc

/**
 * \file BCHelper.h
 */

namespace WasatchCore {
  
  // !!! ACHTUNG !!!
  // !!! READ THE NOMENCLATURE ASSUMPTIONS BEFORE PROCEEDING WITH THIS CLASS !!!
  /* 
     Hi. This class is based on the following assumptions:
   1. We distinguish between boundaries and boundary conditions. 
   2. We associate boundaries with physical domain constraints.
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

  //****************************************************************************
  /**
   *  @struct DomainInfo
   *  @author Tony Saad
   *  @date   Oct 2016
   *
   *  @brief  Stores domain information
   */
  //****************************************************************************
  struct DomainInfo
  {
    double lx, ly, lz;
  };

  
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
    bool operator==(const BndCondSpec& l) const;

    // compare based on the varname only
    bool operator==(const std::string& varNameNew) const;

    // print
    void print() const;

    // check if the user is applying a functor in this boundary condition
    bool is_functor() const;
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
    bool has_patch(const int& patchID) const;
    
    // find the BCSpec associated with a given variable name
    const BndCondSpec* find(const std::string& varName) const;
    
    // find all the BCSpec associated with a given variable name
    std::vector<BndCondSpec> find_all(const std::string& varName) const;
    
    // find the BCSpec associated with a given variable name - non-const version
    const BndCondSpec* find(const std::string& varName);
    
    // check whether this boundary has any bcs specified for varName
    bool has_field(const std::string& varName) const;
    
    // print information about this boundary
    void print() const;
  };
  
  typedef std::map <std::string, BndSpec> BndMapT;
  
  //****************************************************************************
  /**
   *  @struct BoundaryIterators
   *  @author Tony Saad
   *  @date   Sept 2013
   *
   *  @brief  Stores the domain's boundary iterators necessary for setting boundary conditions.
   *  For particles, we store a pointer to a list of particles near a given boundary. That list
   is updated by the ParticlesHelper class. This external list stores the particle index that is
   near the boundary, NOT the particle ID (PID).
   */
  //****************************************************************************
  struct BoundaryIterators
  {
    std::vector<SpatialOps::IntVec> extraBndCells;          // iterator for extra cells. These are zero-based on the extra cell
    std::vector<SpatialOps::IntVec> extraPlusBndCells;      // iterator for extra cells on plus faces (staggered fields). These are zero-based on the extra cell.
    std::vector<SpatialOps::IntVec> interiorBndCells;       // iterator for interior cells. These are zero-based on the extra cell
    
    SpatialOps::SpatialMask<SVolField>* svolExtraCellSpatialMask; // iterator for svol/ccvar extra cells.
    SpatialOps::SpatialMask<XVolField>* xvolExtraCellSpatialMask; // iterator for xvol/sfcxvar extra cells.
    SpatialOps::SpatialMask<YVolField>* yvolExtraCellSpatialMask; // iterator for yvol/sfcyvar extra cells.
    SpatialOps::SpatialMask<ZVolField>* zvolExtraCellSpatialMask; // iterator for zvol/sfczvar extra cells.

    SpatialOps::SpatialMask<SVolField>* svolInteriorCellSpatialMask; // iterator for svol/ccvar interior cells, adjacent to a boundary.
    
    /**
     \brief Helper function to return the appropriate spatial mask given a field type
     */
    template<typename FieldT>
    SpatialOps::SpatialMask<FieldT>*
    get_spatial_mask(bool interior=false) const; // interior is optional and returns the "interior" points adjacent to a boundary. Currently supported for SVOL fields ONLY.

    std::vector<SpatialOps::IntVec> interiorEdgeCells;      // iterator for interior edge (domain edges) cells
    Uintah::Iterator extraBndCellsUintah;                   // We still need the Unitah iterator
    const std::vector<int>* particleIdx;                    // list of particle indices near a given boundary. Given the memory of ALL particles on a given patch, this vector stores
                                                            // the indices of those particles that are near a boundary.
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
    
  protected:
    typedef SpatialOps::IntVec                            IntVecT          ;  // SpatialOps IntVec
    typedef std::map <int, BoundaryIterators            > PatchIDBndItrMapT;  // temporary typedef map that stores boundary iterators per patch id: Patch ID -> Bnd Iterators
    typedef std::map <std::string, PatchIDBndItrMapT    > MaskMapT         ;  // boundary name -> (patch ID -> Boundary iterators )
    
    Uintah::PatchSet* localPatches_;
    const Uintah::MaterialSet* const materials_   ;
    
    // This map stores the iterators associated with each boundary condition name.
    // The iterators are stored in a map keyed by patch ID. a single iterator will be associated
    // with each boundary (aka child)
    MaskMapT                   bndNamePatchIDMaskMap_;
    
    // bndNameBndSpecMap_ stores BndSpec information for each of the specified boundaries. This
    // map is indexed by the (unique) boundary name.
    BndMapT                    bndNameBndSpecMap_;
    
    // Add boundary iterator (mask) for boundary "bndName" and patch "patchID"
    void add_boundary_mask( const BoundaryIterators& myIters,
                            const std::string& bndName,
                            const int& patchID );
    
    // Add a new boundary to the list of boundaries specified for this problem. If the boundary
    // already exists, this means that this boundary is shared by several patches. In that case,
    // add the new patchID to the list of patches that this boundary lives on
    void add_boundary( const std::string&      bndName,
                       Uintah::Patch::FaceType face,
                       const BndTypeEnum&      bndType,
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
    

    BCHelper( const Uintah::LevelP& level,
              Uintah::SchedulerP& sched,
              const Uintah::MaterialSet* const materials );
            
    ~BCHelper();

    /**
     \brief Returns a pointer to the list of particles that are near a given boundary on a given patch ID.
     This pointer is set externally through the ParticlesHelper class.
     */
    const std::vector<int>* get_particles_bnd_mask( const BndSpec& myBndSpec,
                                                   const int& patchID ) const;
    
    /**
     \brief Returns a pointer to the extra cell indices at the specified boundary. The indices correspond
     to the specific FieldType passed to the function template. This covers volume fields ONLY: 
     SVolField, XVolField, YVolField, and ZVolField.
     */
    template<typename FieldT>
    const std::vector<IntVecT>* get_extra_bnd_mask( const BndSpec& myBndSpec,
                                                   const int& patchID ) const;

    /**
     \brief Returns a pointer to the interior cell indices at the specified boundary. The indices correspond
     to the specific FieldType passed to the function template. This covers volume fields ONLY:
     SVolField, XVolField, YVolField, and ZVolField.
     */
    template<typename FieldT>
    const std::vector<IntVecT>* get_interior_bnd_mask( const BndSpec& myBndSpec,
                                                      const int& patchID ) const;
    
    /**
     \brief Returns the extra cell SpatialMask associated with this boundary given the field type.
     This covers volume fields ONLY: SVolField, XVolField, YVolField, and ZVolField.
     */
    template<typename FieldT>
    const SpatialOps::SpatialMask<FieldT>* get_spatial_mask( const BndSpec& myBndSpec,
                                                             const int& patchID,
                                                             const bool interior=false) const;

    /**
     \brief Returns the original Uintah boundary cell iterator.
     */
    Uintah::Iterator& get_uintah_extra_bnd_mask( const BndSpec& myBndSpec,
                                                const int& patchID );

    /**
     \brief Returns the domain edge cells. This will be deprecated in the near future.
     */
    const std::vector<IntVecT>* get_edge_mask( const BndSpec& myBndSpec,
                                              const int& patchID ) const;

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
                                           const double newValue,
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
   
    /**
     *  \brief Retrieve a reference to the boundary and boundary condition information stored in this
     *  BCHelper
     */
    const BndMapT& get_boundary_information() const;

    /**
     *  \brief Returns true of the BCHelper on this patch has any physical boundaries
     */
    bool has_boundaries() const;
    
    /**
     *  \brief Print boundary conditions summary.
     *
     */
    void print() const;
    
  }; // class BCHelper
  
} // namespace WasatchCore

#endif /* defined(WASATCH_BC_HELPER) */
