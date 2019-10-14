/*
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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

#ifndef WBC_HELPER
#define WBC_HELPER

//-- Arches Includes --//
#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/BoundaryConditions/BoundaryFunctorHelper.h>

//-- C++ Includes --//
#include <map>
#include <set>
#include <list>
#include <string>

//-- Uintah Includes --//
#include <Core/Grid/Variables/ComputeSet.h> // used for Uintah::PatchSet
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Grid/Variables/ListOfCellsIterator.h> // used for Uintah::PatchSet
/**
 * \file WBCHelper.h
 */

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

//typedef std::map<std::string, std::set<std::string> > BCFunctorMap;

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
  CUSTOM
};

/**
* @enum BndEdgeType
* @brief If EDGE, then the boundary is defined as a FACE type. If INTERIOR, then
*        the boundary condition is INTERIORFACE type.
*
**/
enum BndEdgeType{
  EDGE,
  INTERIOR
};

BndCondTypeEnum select_bc_type_enum( const std::string& bcTypeStr );
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
  WALL,      ///< Stationary wall BC. Zero velocity (and momentum).
  INLET,     ///< Inlet boundary condition
  OUTLET,    ///< Outlet boundary condition
  PRESSURE,  ///< Pressure boundary condition
  USER,      ///< User specified
  INTRUSION, ///< Intrusion - enum stored here for convenience
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
  VECTOR_TYPE,
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
  std::string      varName;     // name of the variable on which we want to apply a BC
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
  BndEdgeType              edge_type; // Face or interior
  double                   area;      // discrete area of this boundary
  std::vector<int>         patchIDs;  // List of patch IDs that this boundary lives on.
                                      //Note that a boundary is typically split between several patches.
  Uintah::BCGeomBase::ParticleBndSpec particleBndSpec;
  std::vector<BndCondSpec> bcSpecVec; // List of ALL the BCs applied at this boundary

  // returns true if this Boundary has parts of it on patchID
  bool has_patch(const int& patchID) const;

  // find the BCSpec associated with a given variable name
  const BndCondSpec* find(const std::string& varName) const;

  // find the BCSpec associated with a given variable name - non-const version
  BndCondSpec* find_to_edit(const std::string& varName);

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
namespace Uintah{
struct BoundaryIterators
{
  BoundaryIterators(int size) :  extraBndCellsUintah(size) , particleIdx(0){ }
  BoundaryIterators(const BoundaryIterators& copyMe) :  extraBndCellsUintah(copyMe.extraBndCellsUintah) , particleIdx(copyMe.particleIdx){ }

  /**
   \brief Helper function to return the appropriate spatial mask given a field type
   */

  ListOfCellsIterator extraBndCellsUintah;                   // We still need the Unitah iterator
  const std::vector<int>* particleIdx;                    // list of particle indices near a given boundary. Given the memory of ALL particles on a given patch, this vector stores
                                                          // the indices of those particles that are near a boundary.

};
}

//****************************************************************************
/**
 *  \class   WBCHelper
 *  \author  Tony Saad
 *  \date    September, 2013
 *
 *  The WBCHelper class provides a centralized approach to dealing with boundary
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

namespace Uintah {

class WBCHelper {

protected:

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

  // The generic functor information
  // MAP< VAR_NAME, MAP< FACE_NAME, BC_INFORMATION> >
  std::map<std::string, ArchesCore::BoundaryFunctorInformation > m_boundary_functor_info_map;

  // Add a new boundary to the list of boundaries specified for this problem. If the boundary
  // already exists, this means that this boundary is shared by several patches. In that case,
  // add the new patchID to the list of patches that this boundary lives on
  void add_boundary( const std::string&      bndName,
                     Uintah::Patch::FaceType face,
                     const BndTypeEnum&      bndType,
                     const BndEdgeType&      bndEdgeType,
                     const int               patchID,
                     const Uintah::BCGeomBase::ParticleBndSpec);

  // Add boundary iterator (mask) for boundary "bndName" and patch "patchID"
  void add_boundary_mask( const BoundaryIterators& myIters,
                          const std::string& bndName,
                          const int& patchID );

  // apply a boundary condition on a field given another one. here, the srcVarName designates the
  // variable from which we should infer other data
  void add_auxiliary_boundary_condition( const std::string& srcVarName,
                                         BndCondSpec targetBCSpec );

  std::map<std::string, const VarLabel*> m_area_labels;

  void create_new_area_label( const std::string name );

  ProblemSpecP m_arches_spec;

public:

  void delete_area_labels();

  enum Direction {XDIR, YDIR, ZDIR};

  WBCHelper( const Uintah::LevelP& level,
            Uintah::SchedulerP& sched,
            const Uintah::MaterialSet* const materials,
            ProblemSpecP arches_spec );

  ~WBCHelper();

  void sched_computeBCAreaHelper( SchedulerP& sched,
                                  const LevelP& level,
                                  const MaterialSet* matls );

  void computeBCAreaHelper( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const IntVector lo,
                            const IntVector hi );

  void sched_bindBCAreaHelper( SchedulerP& sched,
                                  const LevelP& level,
                                  const MaterialSet* matls );

  void bindBCAreaHelper( const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );

  /**
   \brief Returns the original Uintah boundary cell iterator.
   */
  Uintah::ListOfCellsIterator& get_uintah_extra_bnd_mask( const BndSpec& myBndSpec,
                                               const int& patchID );

  // Parse boundary conditions specified through the input file. This function does NOT need
  // an input file since Uintah already parsed and processed most of this information.
  void parse_boundary_conditions(const int ilvl);


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
   *  WBCHelper
   */
  const BndMapT& get_boundary_information() const;

  /**
   *  \brief Retrieve a reference to the boundary and boundary condition information stored in this
   *  WBCHelper for editing
   */
  BndMapT& get_for_edit_boundary_information();

  /**
   *  \brief Returns true of the WBCHelper on this patch has any physical boundaries
   */
  bool has_boundaries() const;

  /**
   *  \brief Print boundary conditions summary.
   *
   */
  void print() const;

}; // class WBCHelper
} //namespace Uintah
#endif /* defined(WBC_HELPER) */
