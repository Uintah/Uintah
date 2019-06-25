/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

/**
 * @file CCA/Components/PhaseField/Util/Definitions.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Util_Definitions_h
#define Packages_Uintah_CCA_Components_PhaseField_Util_Definitions_h

#ifndef _DOXYGEN
#   define _DOXYGEN 0
#   define _DOXYARG(x)
#endif

#include <sci_defs/kokkos_defs.h>
#include <sci_defs/hypre_defs.h>

#include <Core/Grid/Patch.h>
#include <Core/Grid/Ghost.h>

#ifdef HAVE_HYPRE
#   include <Core/Grid/Variables/Stencil7.h>
#endif

/// @cond DOXYIGNORE
#define TODO { ASSERTFAIL ( "TO BE IMPLEMENTED" ); }
#define VIRT { ASSERTFAIL ( "VIRTUAL METHOD" ); }
/// @endcond

namespace Uintah
{
namespace PhaseField
{

namespace detail
{
template <typename Field> class view;
}

/**
 * @brief Variable Type
 *
 * Enumeration for different types of variable representation
 */
enum VarType : size_t
{
    CC, ///< Cell Centered Variable
    NC, ///< Node Centered Variable
    PP  ///< PerPatch Variable ( used for Problems )
};

/**
 * @brief Dimension Type
 *
 * Enumeration for different problem dimensions
 */
enum DimType : size_t
{
    D1 = 1, ///< 1D Problem
    D2 = 2, ///< 2D Problem
    D3 = 3  ///< 3D Problem
};

/**
 * @brief Stencil Type
 *
 * Enumeration for different finite-difference stencils
 */
enum StnType : size_t
{
    P3, ///< 3 Point stencil for 1D finite-differences
    P5, ///< 5 Point stencil for 2D finite-differences
    P7, ///< 7 Point stencil for 3D finite-differences
    EU  ///< 5 Point stencil for 2D finite-differences as used in campfire
};

/**
 * @brief Direction Type
 *
 * Enumeration for coordinate directions
 */
enum DirType : size_t
{
    X = 0, ///< x Direction
    Y = 1, ///< y Direction
    Z = 2  ///< z Direction
};

/**
 * @brief Interpolation Type
 *
 * Enumeration for different order of interpolations
 */
enum FCIType : size_t
{
    I0 = 1, ///< Piece-wise Interpolation (0th order: 1 point stencil
    I1 = 2  ///< Linear Interpolation in each direction (1st order: 2 DIM points stencil)
};

/// Constant to be used as template parameter to enable adaptive mesh refinement
static constexpr bool AMR { true };

/**
 * @brief Boundary Conditions
 *
 * Enumeration for different types of boundary conditions.
 * @remark They are intended to be used as masks together with FC
 */
enum class BC : size_t
{
    Unknown             = 0x00000000, ///< Unknown condition
    None                = 0x00000100, ///< No condition imposed to the solution \f$ u:\Omega\to\mathbb K^N \f$ over the boundary \f$ \partial\Omega \f$
    Dirichlet           = 0x00000200, ///< Given \f$ a\in\mathbb R \f$ we impose \f$ u = a, u\in\partial\Omega \f$.
    Neumann             = 0x00000300, ///< Given \f$ a\in\mathbb R \f$ we impose \f$ \nabla u \cdot \mathbb n = a, u\in\partial\Omega \f$.
    Symmetry            = 0x00000400, ///< Ghost values for \f$ u \f$ are taken to be equal to the corresponding value within the domain \f$ \Omega \f$
    FineCoarseInterface = 0x00000500  ///< The boundary is an artificial one over a fine level \f$\Omega_k\f$ and continuity has to be imposed across it
};

/**
 * @brief Fine/Coarse Interface Conditions
 *
 * Enumeration for different types of fine/coarse interface conditions.
 * @remark They are intended to be used as masks together with BC
 *
 * Multiple interpolation methods are provided; the following figures show the
 * differences between them in terms of cell used (solid blue) on the fine
 * (diamonds) and coarse (square) levels when computing the value on a ghost
 * cell (red) required by the approximation of the laplacian in 2D with a
 * 5-points stencil across different geometries of fine-coarse interfaces
 *
 * @image latex fc0.eps "FC0"
 * @image html  fc0.png "FC0"
 * @image latex fc1.eps "FC1"
 * @image html  fc1.png "FC1"
 * @image latex fcsimple.eps "FCSimple"
 * @image html  fcsimple.png "FCSimple"
 * @image latex fclinear.eps "FCLinear"
 * @image html  fclinear.png "FCLinear"
 * @image latex fcbilinear.eps "FCBilinear"
 * @image html  fcbilinear.png "FCBilinear"
 */
enum class FC : size_t
{
    None        = 0x00000000,       ///< Continuity across interfaces is not enforced
    FC0         = I0 * 0x00100000,  ///< Piece-wise interpolation is used to compute the value on fine levels ghosts from the old solution at coarser levels @remark to be used with HypreFACSolver @remark should be used with NC variables since no interpolation is required
    FC1         = I1 * 0x00100000,  ///< Linear interpolation in each direction is used to compute the value on fine levels ghosts from the old solution at coarser levels @remark to be used with HypreFACSolver @remark should be used with CC variables since interpolation reduces the approximation error of solution derivatives
#ifdef HAVE_HYPRE
    FCNew       = 0x00010000,       ///< Continuity on fine levels is enforced using the new solution computed at coarser levels
    FC0New      = FC0 + FCNew,      ///< Piece-wise interpolation is used to compute the value on fine levels ghosts from the new solution computed at coarser levels @remark to be used with HypreSolver @remark should be used with NC variables since no interpolation is required
    FC1New      = FC1 + FCNew,      ///< Linear interpolation in each direction is used to compute the value on fine levels ghosts from the new solution computed at coarser levels @remark to be used with HypreSolver @remark should be used with CC variables since interpolation reduces the approximation error of solution derivatives
#endif
    FCSimple    = FC0 + 0x00020000, ///< 2 elements interpolation for 2D problems and CC variables
    FCLinear    = FC0 + 0x00030000, ///< 3 elements extrapolation for 2D problems and CC variables
    FCBilinear  = FC0 + 0x00040000  ///< 4 elements interpolation for 2D problems and CC variables
};

/**
 * @brief Time Discretization Schemes
 *
 * Enumeration for different time discretization schemes.
 * @remark They are intended to be used as masks
 */
enum class TS : size_t
{
    Unknown       = 0x00000000, ///< Unknown time scheme

    Explicit      = 0x01000000, ///< Explicit time schemes mask
    ForwardEuler  = 0x01000001, ///< Explicit forward Euler time scheme

#ifdef HAVE_HYPRE
    SemiImplicit  = 0x02000000, ///< SemiImplicit time schemes mask
    SemiImplicit0 = 0x02000001, ///< SemiImplicit0 time scheme (Application dependent)
    SemiImplicit1 = 0x02000002, ///< SemiImplicit1 time scheme (Application dependent)
    SemiImplicit2 = 0x02000003, ///< SemiImplicit0 time scheme (Application dependent)
    SemiImplicit3 = 0x02000004, ///< SemiImplicit0 time scheme (Application dependent)
    SemiImplicit4 = 0x02000005, ///< SemiImplicit0 time scheme (Application dependent)
    SemiImplicit5 = 0x02000006, ///< SemiImplicit0 time scheme (Application dependent)
    SemiImplicit6 = 0x02000007, ///< SemiImplicit0 time scheme (Application dependent)

    Implicit      = 0x04000000, ///< Implicit time schemes mask
    BackwardEuler = 0x04000001, ///< Implicit backward Euler time scheme
    CrankNicolson = 0x04000002  ///< Implicit Crank-Nicolson time scheme
#endif
};

/**
 * @brief Type for packing BC, FC, and Patch::Face together
 *
 * BCF are used to identify different implementations over patches boundaries
 * which may depend on the type of boundary conditions BC, the type of
 * fine/coarse interface conditions FC and on the actual face on which the
 * implementation is meant to work
 */
using BCF = size_t;

/**
 * @brief VarType Helper
 *
 * Helper struct to get expressions dependent on the given type of variable
 * representation
 * @tparam VAR type of variable representation
 */
template<VarType VAR> struct get_var
#if _DOXYGEN
{
    /// GhostType to be used with given VarType VAR
    static constexpr Ghost::GhostType ghost_type;
}
#endif
;

/**
 * @brief VarType Helper (CC Implementation)
 *
 * CC Implementation of the helper struct to get expressions dependent on the
 * given type of variable representation
 * @implements get_var < VAR >
 */
template<>
struct get_var<CC>
{
    /// GhostType to be used with VAR = CC
    static constexpr Ghost::GhostType ghost_type = Ghost::AroundCells;
};

/**
 * @brief VarType Helper (NC Implementation)
 *
 * NC Implementation of the helper struct to get expressions dependent on the
 * given type of variable representation
 * @implements get_var < VAR >
 */
template<>
struct get_var<NC>
{
    /// GhostType to be used with VAR = NC
    static constexpr Ghost::GhostType ghost_type = Ghost::AroundNodes;
};

/**
 * @brief DimType Helper
 *
 * Helper struct to get expressions dependent on the given problem dimension
 * @tparam DIM problem dimension
 */
template <DimType DIM>
struct get_dim
{
    /// FaceType to be used as first value when forward-iterating over faces
    static constexpr Patch::FaceType face_start = Patch::xminus;

    /// FaceType to be used as lest value when forward-iterating over faces
    static constexpr Patch::FaceType face_end = Patch::FaceType ( 2 * DIM );

    /// Highest valid direction for the given DIM
    static constexpr DirType highest_dir = ( DirType ) ( DIM - 1 );

    /// Unit vector for the given DIM
    /// @return IntVector with relevant entries set to 1 (entries with index greater than DIM are set to 0)
    static IntVector unit_vector()
    {
        return IntVector ( 1, ( DIM > D1 ) ? 1 : 0, ( DIM > D2 ) ? 1 : 0 );
    }

    /// Scalar vector in the given DIM
    /// @tparam V scalar value
    /// @return IntVector with relevant entries set to V (entries with index greater than DIM are set to 0)
    template<int V>
    static IntVector scalar_vector()
    {
        return IntVector ( V, ( DIM > D1 ) ? V : 0, ( DIM > D2 ) ? V : 0 );
    }
};

/**
 * @brief StnType Helper
 *
 * Helper struct to get expressions dependent on the given finite-differences
 * stencil
 * @tparam STN finite-differences stencil
 */
template<StnType STN> struct get_stn
#if _DOXYGEN
{
    /// DimType corresponding to the given stencil
    static constexpr DimType dim;

    /// ghosts required by the  given stencil
    static constexpr int ghosts;

    /// type of the stencil to be used as template parameter for Variable
    template<typename T> using type;
}
#endif
;

/**
 * @brief StnType Helper (P3 implementation)
 *
 * P3 Implementation of the helper struct to get expressions dependent on the
 * given finite-differences stencil
 * @implements get_stn < STN >
 */
template<>
struct get_stn<P3>
{
    /// DimType to be used with STN = P3
    static constexpr DimType dim = D1;

    /// number of ghosts to be used with STN = P3
    static constexpr int ghosts = 1;

#ifdef HAVE_HYPRE
    /// type of the stencil to be used as template parameter for Variable with STN = P3
    template<typename T> using type = typename std::conditional< std::is_same< typename std::remove_const<T>::type , double>::value, Stencil7, void * >::type;
#endif
};

/**
 * @brief StnType Helper (P5 implementation)
 *
 * P5 Implementation of the helper struct to get expressions dependent on the
 * given finite-differences stencil
 * @implements get_stn < STN >
 */
template<>
struct get_stn<P5>
{
    /// DimType to be used with STN = P5
    static constexpr DimType dim = D2;

    /// number of ghosts to be used with STN = P5
    static constexpr int ghosts = 1;

#ifdef HAVE_HYPRE
    /// type of the stencil to be used as template parameter for Variable with STN = P5
    template<typename T> using type = typename std::conditional < std::is_same< typename std::remove_const<T>::type, double>::value, Stencil7, void * >::type;
#endif
};

/**
 * @brief StnType Helper (P7 implementation)
 *
 * P7 Implementation of the helper struct to get expressions dependent on the
 * given finite-differences stencil
 * @implements get_stn < STN >
 */
template<>
struct get_stn<P7>
{
    /// DimType to be used with STN = P7
    static constexpr DimType dim = D3;

    /// number of ghosts to be used with STN = P7
    static constexpr int ghosts = 1;

#ifdef HAVE_HYPRE
    /// type of the stencil to be used as template parameter for Variable with STN = P7
    template<typename T> using type = typename std::conditional <  std::is_same< typename std::remove_const<T>::type, double>::value, Stencil7, void * >::type;
#endif
};

/**
 * @brief StnType Helper (EU implementation)
 *
 * EU Implementation of the helper struct to get expressions dependent on the
 * given finite-differences stencil
 * @implements get_stn < STN >
 */
template<>
struct get_stn<EU>
{
    /// DimType to be used with STN = EU
    static constexpr DimType dim = D2;

    /// number of ghosts to be used with STN = EU
    static constexpr int ghosts = 1;

#ifdef HAVE_HYPRE
    /// type of the stencil to be used as template parameter for Variable with STN = EU
    template<typename T> using type = typename std::conditional < std::is_same< typename std::remove_const<T>::type, double>::value, Stencil7, void >::type;
#endif
};

/**
 * @brief DirType Helper
 *
 * Helper struct to get expressions dependent on the given coordinate direction
 */
template <DirType DIR>
struct get_dir
{
    /// DirType corresponding to the coordinate direction whose index precedes DIR
    static constexpr DirType lower = ( DirType ) ( DIR - 1 );

    /// DirType corresponding to the coordinate direction whose index follows DIR
    static constexpr DirType higher = ( DirType ) ( DIR + 1 );

    /// Patch::FaceType orthogonal to DIR (lower face)
    static constexpr Patch::FaceType minus_face = ( Patch::FaceType ) ( 2 * DIR );

    /// Patch::FaceType orthogonal to DIR (upper face)
    static constexpr Patch::FaceType plus_face = ( Patch::FaceType ) ( 2 * DIR + 1 );
};

/// @cond DOXYIGNORE
template <DirType DIR> constexpr Patch::FaceType get_dir<DIR>::minus_face;
template <DirType DIR> constexpr Patch::FaceType get_dir<DIR>::plus_face;
/// @endcond

/**
 * @brief FCIType Helper
 *
 * Helper struct to get expressions dependent on the given order of
 * interpolation
 * @tparam FCI order of interpolation
 */
template <FCIType FCI> struct get_fci
#if _DOXYGEN
{
    /// number of cells/nodes required for interpolation
    static constexpr int elems;
}
#endif
;

/**
 * @brief FCIType Helper (I0 Implementation)
 *
 * I0 Implementation of the helper struct to get expressions dependent on the
 * given order of interpolation
 * @implements get_fci < FCI >
 */
template <>
struct get_fci<I0>
{
    /// number of cells/nodes required for interpolation
    static constexpr int elems = 1;
};

/**
 * @brief FCIType Helper (I1 Implementation)
 *
 * I1 Implementation of the helper struct to get expressions dependent on the
 * given order of interpolation
 * @implements get_fci < FCI >
 */
template <>
struct get_fci<I1>
{
    /// number of cells/nodes required for interpolation/restriction
    static constexpr int elems = 2;
};

/**
 * @brief Patch::FaceType Helper
 *
 * Helper struct to get expressions dependent on the given face
 * @tparam Patch::FaceType face
 */
template <Patch::FaceType F>
struct get_face
{
    /// face direction
    static constexpr DirType dir = ( DirType ) ( F / 2 );

    /// face sign (int)
    static constexpr int sgn = 2 * ( F % 2 ) - 1;

    /// face sign (double)
    static constexpr double dsgn = static_cast<double> ( sgn );
};

/**
 * @brief BC Helper
 *
 * Helper struct to get expressions dependent on the given type of boundary
 * conditions
 *
 * @tparam B type of boundary conditions
 */
template<BC B> struct get_bc
#if _DOXYGEN
{
    /// @brief type of bc rhs for the given type of boundary conditions
    /// Is the type used for handling the rhs in the boundary condition equation
    /// @tparam Field field type of the variable on which the condition is applied
    template<typename Filed> using value_type;
}
#endif
;

/**
 * @brief BC Helper (Dirichlet implementation)
 *
 * Dirichlet implementation of the helper struct to get expressions dependent on
 * the given type of boundary conditions
 * @implements get_bc < BC >
 */
template<>
struct get_bc<BC::Dirichlet>
{
    /// @brief type of bc rhs for Dirichlet boundary conditions
    /// for Dirichlet condition on a scalar variable (T=double, const double)
    /// value_type = double; otherwise value_type = void * (undefined)
    /// @tparam Field field type of the variable on which the condition is applied
    template<typename Field> using value_type = typename Field::value_type;
};

/**
 * @brief BC Helper (Neumann implementation)
 *
 * Neumann implementation of the helper struct to get expressions dependent on
 * the given type of boundary conditions
 * @implements get_bc < BC >
 */
template<>
struct get_bc<BC::Neumann>
{
    /// @brief type of bc rhs for Neumann boundary conditions
    /// for Neumann condition on a scalar variable (T=double, const double)
    /// value_type = double; otherwise value_type = void * (undefined)
    /// @tparam Field field type of the variable on which the condition is applied
    template<typename Field> using value_type = typename Field::value_type;
};

/**
 * @brief BC Helper (FineCoarseInterface implementation)
 *
 * FineCoarseInterface implementation of the helper struct to get expressions
 * dependent on the given type of boundary conditions
 * @implements get_bc < BC >
 */
template<>
struct get_bc<BC::FineCoarseInterface>
{
    /// @brief type of bc rhs for FineCoarseInterface boundary conditions
    /// for FineCoarseInterface condition on a scalar variable (T=double,
    /// const double) value_type = V *; otherwise value_type = void *
    /// (undefined)
    /// @tparam Field field type of the variable on which the condition is applied
    template<typename Field> using value_type = detail::view<Field> * ;
};

/**
 * @brief FC Helper
 *
 * Helper struct to get expressions dependent on the given type of fine/coarse
 * interface conditions
 *
 * @tparam F type of fine/coarse interface conditions
 */
template < FC F > struct get_fc
{
    /// order of interpolation
    static constexpr FCIType fci = ( FCIType ) ( ( size_t ) F / 0x00100000 );
};

/**
 * @brief BCF Helper
 *
 * Helper struct to get expressions dependent on the given BC, FC, and
 * Patch::Face pack
 *
 * @tparam P BC, FC, and Patch::Face pack
 */
template < BCF P >
struct get_bcf
{
    /// type of boundary conditions
    static constexpr BC bc = ( BC ) ( P & 0x0000FF00 );

    /// type of fine/coarse interface conditions
    static constexpr FC c2f = ( FC ) ( P & 0x00FF0000 );

    /// patch face
    static constexpr Patch::FaceType face = ( Patch::FaceType ) ( P & 0x000000FF );
};

/**
 * @brief VarType to std::string
 *
 * Converts the type of variable representation to the corresponding string as
 * used by factory constructors
 * @param var value to serialize
 * @return serialized string
 */
inline std::string
var_to_str (
    VarType var
)
{
    switch ( var )
    {
    case CC:
        return "CC";
        break;
    case NC:
        return "NC";
        break;
    default:
        return "Unknown";
    }
}

/**
 * @brief DimType to std::string
 *
 * Converts the problem dimension to the corresponding string as used by factory
 * constructors
 * @param dim value to serialize
 * @return serialized string
 */
inline std::string
dim_to_str (
    DimType dim
)
{
    switch ( dim )
    {
    case D1:
        return "D1";
    case D2:
        return "D2";
    case D3:
        return "D3";
    default:
        return "Unknown";
    }
}

/**
 * @brief StnType to std::string
 *
 * Converts the finite-difference stencil to the corresponding string as used by
 * factory constructors
 * @param stn value to serialize
 * @return serialized string
 */
inline std::string
stn_to_str (
    StnType stn
)
{
    switch ( stn )
    {
    case P3:
        return "P3";
    case P5:
        return "P5";
    case P7:
        return "P7";
    case EU:
        return "EU";
    default:
        return "Unknown";

    }
}

/**
 * @brief DirType to std::string
 *
 * Converts the type of variable representation to the corresponding string as
 * used by factory constructors
 * @param dir value to serialize
 * @return serialized string
 */
inline std::string
dir_to_str (
    DirType dir
)
{
    switch ( dir )
    {
    case X:
        return "X";
    case Y:
        return "Y";
    case Z:
        return "Z";
    default:
        return "Unknown";
    }
}

/**
 * @brief BC to std::string
 *
 * Converts the type of boundary conditions to the corresponding string as
 * used by factory constructors
 * @param bc value to serialize
 * @return serialized string
 */
inline std::string
bc_to_str (
    BC bc
)
{
    switch ( bc )
    {
    case BC::None:
        return "None";
    case BC::Dirichlet:
        return "Dirichlet";
    case BC::Neumann:
        return "Neumann";
    case BC::Symmetry:
        return "Symmetry";
    case BC::FineCoarseInterface:
        return "FineCoarseInterface";
    default:
        return "Unknown";
    }
}

/**
 * @brief FC to std::string
 *
 * Converts the type of fine/coarse interface conditions to the corresponding
 * string as used by factory constructors
 * @param fc value to serialize
 * @return serialized string
 */
inline std::string
fc_to_str (
    FC fc
)
{
    switch ( fc )
    {
    case FC::None:
        return "None";
    case FC::FC0:
        return "FC0";
    case FC::FCSimple:
        return "FCSimple";
    case FC::FCLinear:
        return "FCLinear";
    case FC::FCBilinear:
        return "FCBilinear";
    case FC::FC1:
        return "FC1";
#ifdef HAVE_HYPRE
    case FC::FC0New:
        return "FC0New";
    case FC::FC1New:
        return "FC1New";
#endif
    default:
        return "Unknown";
    }
}

/**
 * @brief std::string to BC
 *
 * Converts the string as used in input files to the corresponding type of
 * boundary conditions
 * @param value value to deserialize
 * @return parsed value
 */
inline BC
str_to_bc (
    const std::string & value
)
{
    if ( value == "Dirichlet" ) return BC::Dirichlet;
    if ( value == "Neumann" ) return BC::Neumann;
    return BC::Unknown;
}

/**
 * @brief std::string to FC
 *
 * Converts the string as used in input files to the corresponding type of
 * fine/coarse interface condition
 * @param value value to deserialize
 * @return parsed value
 */
inline FC
str_to_fc (
    const std::string & value
)
{
    if ( value == "FC0" ) return FC::FC0;
    if ( value == "FCSimple" ) return FC::FCSimple;
    if ( value == "FCLinear" ) return FC::FCLinear;
    if ( value == "FCBilinear" ) return FC::FCBilinear;
    if ( value == "FC1" ) return FC::FC1;
#ifdef HAVE_HYPRE
    if ( value == "FC0New" ) return FC::FC0New;
    if ( value == "FC1New" ) return FC::FC1New;
#endif
    return FC::None;
}

/**
 * @brief std::string to TS
 *
 * Converts the string as used in input files to the corresponding time
 * discretization scheme
 * @param value value to deserialize
 * @return parsed value
 */
inline TS
str_to_ts (
    const std::string & value
)
{
    if ( value == "forward_euler" )   return TS::ForwardEuler;
#ifdef HAVE_HYPRE
    if ( value == "semi_implicit_0" ) return TS::SemiImplicit0;
    if ( value == "semi_implicit_1" ) return TS::SemiImplicit1;
    if ( value == "semi_implicit_2" ) return TS::SemiImplicit2;
    if ( value == "semi_implicit_3" ) return TS::SemiImplicit3;
    if ( value == "semi_implicit_4" ) return TS::SemiImplicit4;
    if ( value == "semi_implicit_5" ) return TS::SemiImplicit5;
    if ( value == "semi_implicit_6" ) return TS::SemiImplicit6;
    if ( value == "backward_euler" )  return TS::BackwardEuler;
    if ( value == "crank_nicolson" )  return TS::CrankNicolson;
#endif
    return TS::Unknown;
}

/**
 * @brief Check FC masks
 *
 * Evaluate if a given FC matches a required type (or category) of fine/coarse
 * interface conditions
 * @param a TS to check
 * @param b TS required type or category of fine/coarse interface conditions
 * @return check result
 */
constexpr bool
operator & (
    FC a,
    FC b
)
{
    return ( size_t ) a & size_t ( b );
}

/**
 * @brief Check TS masks
 *
 * Evaluate if a given TS matches a required scheme or type of scheme
 * @param a TS to check
 * @param b TS required scheme or type of scheme
 * @return check result
 */
constexpr bool
operator & (
    TS a,
    TS b
)
{
    return ( size_t ) a & size_t ( b );
}

/**
 * @brief Add FC to masks
 *
 * Set FC bits of the given mask
 * @tparam A mask type
 * @param a given mask
 * @param b FC to set
 * @return new BCF mask
 */
template<typename A>
constexpr BCF
operator | (
    A a,
    FC b
)
{
    return ( BCF ) a | ( BCF ) ( b );
}

/**
 * @brief Add BC to masks
 *
 * Set BC bits of the given mask
 * @tparam A mask type
 * @param a given mask
 * @param b BC to set
 * @return new BCF mask
 */
template<typename A>
constexpr BCF
operator | (
    A a,
    BC b
)
{
    return ( BCF ) a | ( BCF ) ( b );
}

/**
 * @brief Add Patch::FaceType to masks
 *
 * Set Patch::FaceType bits of the given mask
 * @tparam A mask type
 * @param a given mask
 * @param b Patch::FaceType to set
 * @return new BCF mask
 */
template<typename A>
constexpr BCF
operator | (
    A a,
    Patch::FaceType b
)
{
    return ( BCF ) a | ( BCF ) ( b );
}

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Util_Definitions_h
