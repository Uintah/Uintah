/*
 * The MIT License
 *
 * Copyright (c) 2013-2015 The University of Utah
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

#include "BCHelper.h"

//-- C++ Includes --//
#include <vector>
#include <iostream>

#include <boost/foreach.hpp>

//-- Uintah Includes --//
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/CellIterator.h> // SCIRun::Iterator
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <CCA/Components/Wasatch/Expressions/Pressure.h>
#include <CCA/Components/Wasatch/Expressions/NullExpression.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/ParticlesHelper.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>

//-- Debug Stream --//
#include <Core/Util/DebugStream.h>

static SCIRun::DebugStream dbgbc("WASATCH_BC", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBC  if( DBC_BC_ON  ) dbgbc

/**
 * \file    BCHelper.cc
 * \author  Tony Saad
 */

namespace Wasatch {
  // Given a string BC type (Dirichlet, Neumann,...), this function returns a BndCondTypeEnum
  // of supported boundary condition types
  BndCondTypeEnum select_bc_type_enum( const std::string& bcTypeStr )
  {
    if      ( bcTypeStr == "Dirichlet" )     return DIRICHLET;
    else if ( bcTypeStr == "Neumann" )       return NEUMANN;
    else                                     return UNSUPPORTED;
  }

  // Given a string boundary type (Wall, Velocity, Outflow,...), this function returns a BndTypeEnum
  // of supported boundary types
  BndTypeEnum select_bnd_type_enum( const std::string& bndTypeStr )
  {
    if      ( bndTypeStr == "Wall"     )  return WALL;
    else if ( bndTypeStr == "Velocity" )  return VELOCITY;
    else if ( bndTypeStr == "Open"     )  return OPEN;
    else if ( bndTypeStr == "Outflow"  )  return OUTFLOW;
    else if ( bndTypeStr == "None"
              || bndTypeStr == "User"  )  return USER;
    else                                  return INVALID;
  }

  // Given a BndCondTypeEnum (DIRICHLET,...), this function returns a string
  // of supported boundary condition types
  std::string bc_type_enum_to_string( const BndCondTypeEnum bcTypeEnum )
  {
    switch (bcTypeEnum) {
      case DIRICHLET:
        return "Dirichlet";
        break;
      case NEUMANN:
        return "Neumann";
        break;
      default:
        return "Unsupported";
        break;
    }
  }

  // Given a BndTypeEnum (WALL, VELOCITY,...), this function returns a string
  // of supported boundary types
  const std::string bnd_type_enum_to_string( const BndTypeEnum bndTypeEnum )
  {
    switch (bndTypeEnum) {
      case WALL:
        return "Wall";
        break;
      case VELOCITY:
        return "Velocity";
        break;
      case OPEN:
        return "Open";
        break;
      case OUTFLOW:
        return "Outflow";
        break;
      case USER:
        return "User";
        break;
      default:
        return "Invalid";
        break;
    }
  }

  template<typename OST>
  OST& operator<<( OST& os, const BndTypeEnum bndTypeEnum )
  {
    os << bnd_type_enum_to_string(bndTypeEnum);
    return os;
  }
  
  template<typename OST>
  OST& operator<<( OST& os, const BndCondTypeEnum bcTypeEnum )
  {
    os << bc_type_enum_to_string(bcTypeEnum);
    return os;
  }  

  // This function returns true if the boundary condition is applied in the same direction
  // as the staggered field. For example, xminus/xplus on a XVOL field.
  template <typename FieldT>
  bool is_staggered_normal( const Uintah::Patch::FaceType face ){
    const Direction staggeredLocation = get_staggered_location<FieldT>();
    switch (staggeredLocation) {
      case XDIR:
        return ( (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus)); break;
      case YDIR:
        return ( (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus)); break;
      case ZDIR:
        return ( (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus)); break;
      default: return false; break;
    }
    return false;
  }
  
  // This function returns true if the boundary condition is applied in the same direction
  // as the staggered field. For example, xminus/xplus on a XVOL field.
  template<typename FieldT>
  bool is_plus_side( const Uintah::Patch::FaceType face ){
    const Direction staggeredLocation = get_staggered_location<FieldT>();
    switch (staggeredLocation) {
      case XDIR: return (face==Uintah::Patch::xplus);  break;
      case YDIR: return (face==Uintah::Patch::yplus);  break;
      case ZDIR: return (face==Uintah::Patch::zplus);  break;
      default:   return false; break;
    }
    return false;
  }

  //============================================================================

  bool BndCondSpec::operator==(const BndCondSpec& l) const
  {
    return (   l.varName == varName
            && l.functorName == functorName
            && l.value == value
            && l.bcType == bcType
            && l.bcValType == bcValType);
  };

  bool BndCondSpec::operator==(const std::string& varNameNew) const
  {
    return ( varNameNew == varName);
  };

  void BndCondSpec::print() const
  {
    using namespace std;
    cout << "  var:     " << varName << endl
         << "  type:    " << bcType << endl
         << "  value:   " << value << endl;
    if( !functorName.empty() )
      cout << "  functor: " << functorName << endl;
  };

  bool BndCondSpec::is_functor() const
  {
    return (bcValType == FUNCTOR_TYPE);
  };

  //============================================================================

  // returns true if this Boundary has parts of it on patchID
  bool BndSpec::has_patch(const int& patchID) const
  {
    return std::find(patchIDs.begin(), patchIDs.end(), patchID) != patchIDs.end();
  }

  // find the BCSpec associated with a given variable name
  const BndCondSpec* BndSpec::find(const std::string& varName) const
  {
    std::vector<BndCondSpec>::const_iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
    if (it != bcSpecVec.end()) {
      return &(*it);
    } else {
      return NULL;
    }
  }

  // find the BCSpec associated with a given variable name - non-const version
  const BndCondSpec* BndSpec::find(const std::string& varName)
  {
    std::vector<BndCondSpec>::iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
    if (it != bcSpecVec.end()) {
      return &(*it);
    } else {
      return NULL;
    }
  }

  // check whether this boundary has any bcs specified for varName
  bool BndSpec::has_field(const std::string& varName) const
  {
    std::vector<BndCondSpec>::const_iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
    if (it != bcSpecVec.end()) {
      return true;
    } else {
      return false;
    }
  }

  // print information about this boundary
  void BndSpec::print() const
  {
    using namespace std;
    cout << "Boundary: " << name << " face: " << face << " BndType: " << type << endl;
    for (vector<BndCondSpec>::const_iterator it=bcSpecVec.begin(); it != bcSpecVec.end(); ++it) {
      (*it).print();
    }
  }

  //============================================================================


  //****************************************************************************
  /**
   *
   *  \brief Computes the coefficients associated with a given boundary condition
   *
   */
  //****************************************************************************

  template <typename FieldT>
  void get_bcop_coefs( const SpatialOps::OperatorDatabase& opdb,
                       const BndSpec& myBndSpec,
                       const BndCondSpec& myBndCondSpec,
                       double& ci,
                       double& cg,
                       bool setOnExtraOnly=false )
  {
    const BndCondTypeEnum& atomBCTypeEnum = myBndCondSpec.bcType;
    const Uintah::Patch::FaceType& face = myBndSpec.face;
    const bool isStagNorm = is_staggered_normal<FieldT>(face);

    if ( setOnExtraOnly || ( isStagNorm && atomBCTypeEnum != NEUMANN ) )
    {
      cg = 1.0;
      ci = 0.0;
      return;
    }
    
    typedef BCOpTypeSelector<FieldT> OpT;
    switch (atomBCTypeEnum) {
      case DIRICHLET:
      {
        switch (face) {
          case Uintah::Patch::xminus: {
            const typename OpT::DirichletX* const op = opdb.retrieve_operator<typename OpT::DirichletX>();
            cg = op->coefs().get_coef(0); //low coefficient
            ci = op->coefs().get_coef(1); //high coefficient
            break;
          }
          case Uintah::Patch::xplus: {
            const typename OpT::DirichletX* const op = opdb.retrieve_operator<typename OpT::DirichletX>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }
          case Uintah::Patch::yminus: {
            const typename OpT::DirichletY* const op = opdb.retrieve_operator<typename OpT::DirichletY>();
            cg = op->coefs().get_coef(0); //low coefficient
            ci = op->coefs().get_coef(1); //high coefficient
            break;
          }
          case Uintah::Patch::yplus: {
            const typename OpT::DirichletY* const op = opdb.retrieve_operator<typename OpT::DirichletY>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }
          case Uintah::Patch::zminus: {
            const typename OpT::DirichletZ* const op = opdb.retrieve_operator<typename OpT::DirichletZ>();
            cg = op->coefs().get_coef(1); //low coefficient
            ci = op->coefs().get_coef(0); //high coefficient
            break;
          }
          case Uintah::Patch::zplus: {
            const typename OpT::DirichletZ* const op = opdb.retrieve_operator<typename OpT::DirichletZ>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }

          default:
          {
            std::ostringstream msg;
            msg << "ERROR: An invalid uintah face has been specified when tyring to apply bc on "
                << myBndCondSpec.varName << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          }
        }
        break; // DIRICHLET
      }

      case NEUMANN:
      {
        switch (face) {
          case Uintah::Patch::xminus: {
            const typename OpT::NeumannX* const op = opdb.retrieve_operator<typename OpT::NeumannX>();
            cg = op->coefs().get_coef(0); //low coefficient
            ci = op->coefs().get_coef(1); //high coefficient
            break;
          }
          case Uintah::Patch::xplus: {
            const typename OpT::NeumannX* const op = opdb.retrieve_operator<typename OpT::NeumannX>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }
          case Uintah::Patch::yminus: {
            const typename OpT::NeumannY* const op = opdb.retrieve_operator<typename OpT::NeumannY>();
            cg = op->coefs().get_coef(0); //low coefficient
            ci = op->coefs().get_coef(1); //high coefficient
            break;
          }
          case Uintah::Patch::yplus: {
            const typename OpT::NeumannY* const op = opdb.retrieve_operator<typename OpT::NeumannY>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }
          case Uintah::Patch::zminus: {
            const typename OpT::NeumannZ* const op = opdb.retrieve_operator<typename OpT::NeumannZ>();
            cg = op->coefs().get_coef(0); //low coefficient
            ci = op->coefs().get_coef(1); //high coefficient
            break;
          }
          case Uintah::Patch::zplus: {
            const typename OpT::NeumannZ* const op = opdb.retrieve_operator<typename OpT::NeumannZ>();
            cg = op->coefs().get_coef(1); //high coefficient
            ci = op->coefs().get_coef(0); //low coefficient
            break;
          }
          default:
          {
            std::ostringstream msg;
            msg << "ERROR: An invalid uintah face has been specified when tyring to apply bc on "
            << myBndCondSpec.varName << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
          }
        }
        break; // NEUMANN
      }
        
      default:
      {
        std::ostringstream msg;
        msg << "ERROR: It looks like you have specified an UNSUPPORTED basic boundary Type!"
        << "Basic boundary types can only be either DIRICHLET or NEUMANN. Please revise your input file." << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
  }

  //****************************************************************************
  /**
   *
   *  \brief Helps with staggered fields.
   *
   */
  //****************************************************************************
  
  void pack_uintah_iterator_as_spatialops( const Uintah::Patch::FaceType& face,
                                           const Uintah::Patch* const patch,
                                           Uintah::Iterator& bndIter,
                                           BoundaryIterators& myBndIters )
  {
    using SpatialOps::IntVec;
    
    std::vector<IntVec>& extraBndSOIter        = myBndIters.extraBndCells;
    std::vector<IntVec>& intBndSOIter          = myBndIters.interiorBndCells;
    std::vector<IntVec>& extraPlusBndCells     = myBndIters.extraPlusBndCells;
    
    std::vector<IntVec> neboExtraBndSOIter;

    std::vector<IntVec>& intEdgeSOIter         = myBndIters.interiorEdgeCells;
   
    bool plusEdge[3];
    bool minusEdge[3];

    minusEdge[0] = patch->getBCType(Uintah::Patch::xminus) != Uintah::Patch::Neighbor;
    plusEdge [0] = patch->getBCType(Uintah::Patch::xplus ) != Uintah::Patch::Neighbor;
    minusEdge[1] = patch->getBCType(Uintah::Patch::yminus) != Uintah::Patch::Neighbor;
    plusEdge [1] = patch->getBCType(Uintah::Patch::yplus ) != Uintah::Patch::Neighbor;
    minusEdge[2] = patch->getBCType(Uintah::Patch::zminus) != Uintah::Patch::Neighbor;
    plusEdge [2] = patch->getBCType(Uintah::Patch::zplus ) != Uintah::Patch::Neighbor;

    int i=-1, j=-1;
    switch (face) {
      case Uintah::Patch::xminus:
      case Uintah::Patch::xplus: i=1; j=2; break;
      case Uintah::Patch::yminus:
      case Uintah::Patch::yplus: i=0; j=2; break;
      case Uintah::Patch::zminus:
      case Uintah::Patch::zplus: i=0; j=1; break;
      default:{
        std::ostringstream msg;
        msg << "ERROR: invalid face specification encountered\n"
            << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }            
    
    // save pointer to the Uintah iterator. This will be needed for expression that require access to the
    // native uintah iterators, such as the pressure expression.
    myBndIters.extraBndCellsUintah = bndIter;

    DBGBC << "---------------------------------------------------\n";
    DBGBC << "Face = " << face << std::endl;
    
    // MAJOR WARNING HERE - WHEN WE MOVE TO RUNTIME GHOST CELLS, WE NEED TO USE THE APPROPRIATE PATCH OFFSET
    const Uintah::IntVector patchCellOffset = patch->getExtraCellLowIndex(1);
    const Uintah::IntVector interiorPatchCellOffset = patch->getCellLowIndex();
    Uintah::IntVector unitNormal = patch->faceDirection(face); // this is needed to construct interior cells
    Uintah::IntVector bcPointIJK;
    
    Uintah::IntVector edgePoint;
    const Uintah::IntVector idxHi = patch->getCellHighIndex() - IntVector(1,1,1);// - patchCellOffset;
    const Uintah::IntVector idxLo = patch->getCellLowIndex();

    for( bndIter.reset(); !bndIter.done(); ++bndIter ){
      bcPointIJK = *bndIter - patchCellOffset;
      extraBndSOIter.push_back(IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
      
      edgePoint = *bndIter - unitNormal;
      if( ((edgePoint[i] == idxHi[i]) && plusEdge[i]  ) ||
          ((edgePoint[j] == idxHi[j]) && plusEdge[j]  ) ||
          ((edgePoint[i] == idxLo[i]) && minusEdge[i] ) ||
          ((edgePoint[j] == idxLo[j]) && minusEdge[j] ) )
      {
        edgePoint -= patchCellOffset;
        intEdgeSOIter.push_back( IntVec(bcPointIJK[0], bcPointIJK[1], bcPointIJK[2]) );
      }
      bcPointIJK -= unitNormal;
      intBndSOIter.push_back(IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));

      bcPointIJK = *bndIter - interiorPatchCellOffset;
      neboExtraBndSOIter.push_back(IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
    }
    
    // if we are on a plus face, we will most likely need a plus-face iterator for staggered fields
    if (face == Uintah::Patch::xplus || face == Uintah::Patch::yplus || face == Uintah::Patch::zplus ){
      for( bndIter.reset(); !bndIter.done(); ++bndIter ){
        bcPointIJK = *bndIter - patchCellOffset + unitNormal;
        extraPlusBndCells.push_back(IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
      }
    }
    
    // convert the svol extra cell boundary iterator to a spatial mask
    const int ng = get_n_ghost<SVolField>();
    SpatialOps::IntVec bcMinus, bcPlus;
    Wasatch::get_bc_logicals( patch, bcMinus, bcPlus );
    SpatialOps::BoundaryCellInfo bcInfo = SpatialOps::BoundaryCellInfo::build<SVolField>(bcPlus);
    SpatialOps::GhostData gd(ng);
    const SpatialOps::MemoryWindow window = Wasatch::get_memory_window_for_masks<SVolField>( patch );
    myBndIters.svolExtraCellSpatialMask = new SpatialOps::SpatialMask<SVolField>(window, bcInfo, gd, neboExtraBndSOIter);
#     ifdef HAVE_CUDA
    myBndIters.svolExtraCellSpatialMask->add_consumer(GPU_INDEX);
#     endif
    
    const SpatialOps::MemoryWindow xwindow = Wasatch::get_memory_window_for_masks<XVolField>( patch );
    SpatialOps::BoundaryCellInfo xBCInfo = SpatialOps::BoundaryCellInfo::build<XVolField>(bcPlus);
    myBndIters.xvolExtraCellSpatialMask = new SpatialOps::SpatialMask<XVolField>(xwindow, xBCInfo, gd, neboExtraBndSOIter);
#     ifdef HAVE_CUDA
    myBndIters.xvolExtraCellSpatialMask->add_consumer(GPU_INDEX);
#     endif
    
    const SpatialOps::MemoryWindow ywindow = Wasatch::get_memory_window_for_masks<YVolField>( patch );
    SpatialOps::BoundaryCellInfo yBCInfo = SpatialOps::BoundaryCellInfo::build<YVolField>(bcPlus);
    myBndIters.yvolExtraCellSpatialMask = new SpatialOps::SpatialMask<YVolField>(ywindow, yBCInfo, gd, neboExtraBndSOIter);
#     ifdef HAVE_CUDA
    myBndIters.yvolExtraCellSpatialMask->add_consumer(GPU_INDEX);
#     endif
    
    const SpatialOps::MemoryWindow zwindow = Wasatch::get_memory_window_for_masks<ZVolField>( patch );
    SpatialOps::BoundaryCellInfo zBCInfo = SpatialOps::BoundaryCellInfo::build<ZVolField>(bcPlus);
    myBndIters.zvolExtraCellSpatialMask = new SpatialOps::SpatialMask<ZVolField>(zwindow, zBCInfo, gd, neboExtraBndSOIter);
#     ifdef HAVE_CUDA
    myBndIters.zvolExtraCellSpatialMask->add_consumer(GPU_INDEX);
#     endif
  }

  
  //============================================================================
  // Template specialization for the BoundaryIterators get_spatial_mask function
  template<typename FieldT>
  SpatialOps::SpatialMask<FieldT>*
  BoundaryIterators::get_spatial_mask() const
  {
    // should never request a spatial_mask of an unsupported type. Supported types are SVol, X-Y-ZVol
    std::ostringstream msg;
    msg << "ERROR: It looks like you were trying to retrieve a spatial mask of an unsupported field type. "
    << " Supported types are: SVol, XVol, YVol, and ZVol." << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    return NULL;
  }

  template<>
  SpatialOps::SpatialMask<SVolField>*
  BoundaryIterators::get_spatial_mask() const
  {
    return svolExtraCellSpatialMask;
  }
  
  template<>
  SpatialOps::SpatialMask<XVolField>*
  BoundaryIterators::get_spatial_mask() const
  {
    return xvolExtraCellSpatialMask;
  }
  
  template<>
  SpatialOps::SpatialMask<YVolField>*
  BoundaryIterators::get_spatial_mask() const
  {
    return yvolExtraCellSpatialMask;
  }
  
  template<>
  SpatialOps::SpatialMask<ZVolField>*
  BoundaryIterators::get_spatial_mask() const
  {
    return zvolExtraCellSpatialMask;
  }
  
  template<>
  SpatialOps::SpatialMask<ParticleField>*
  BoundaryIterators::get_spatial_mask() const
  {
    return NULL;
  }

  //============================================================================

  //************************************************************************************************
  //
  //                          IMPLEMENTATION
  //
  //************************************************************************************************
  
  //------------------------------------------------------------------------------------------------

  BCHelper::BCHelper( const Uintah::PatchSet* const localPatches,
                      const Uintah::MaterialSet* const materials,
                      const PatchInfoMap& patchInfoMap,
                      GraphCategories& grafCat,
                      BCFunctorMap& bcFunctorMap )
  : localPatches_(localPatches),
    materials_   (materials   ),
    patchInfoMap_(patchInfoMap),
    bcFunctorMap_(bcFunctorMap),
    grafCat_     (grafCat)
  {
    parse_boundary_conditions();
  }

  //------------------------------------------------------------------------------------------------

  BCHelper::~BCHelper()
  {
    typedef MaskMapT::iterator it_type;
    typedef PatchIDBndItrMapT::iterator secondIt;
    for(it_type iterator = bndNamePatchIDMaskMap_.begin(); iterator != bndNamePatchIDMaskMap_.end(); iterator++) {
      PatchIDBndItrMapT pidBndItr = iterator->second;
      for(secondIt sit = pidBndItr.begin(); sit != pidBndItr.end(); sit++) {
        delete sit->second.svolExtraCellSpatialMask;
        delete sit->second.xvolExtraCellSpatialMask;
        delete sit->second.yvolExtraCellSpatialMask;
        delete sit->second.zvolExtraCellSpatialMask;
      }
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary_condition( const std::string& bndName,
                                         const BndCondSpec& bcSpec )
  {
    using namespace std;
    if ( bndNameBndSpecMap_.find(bndName) != bndNameBndSpecMap_.end() ) {
      BndSpec& existingBCSpec = (*bndNameBndSpecMap_.find(bndName)).second;
      vector<BndCondSpec>& bcSpecVec = existingBCSpec.bcSpecVec;
      vector<BndCondSpec>::iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), bcSpec);
      if ( it == bcSpecVec.end() ) {
        DBGBC << "adding bc " << bcSpec.varName << " on " << bndName << " \n";
        bcSpecVec.push_back(bcSpec);
      } else {
        DBGBC << "bc " << bcSpec.varName << " already exists on " << bndName << ". skipping \n";
      }
    } else {
      DBGBC << " ERROR! boundary face " << bndName << " does not exist!!! \n";
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary_condition( const BndCondSpec& bcSpec )
  {
    using namespace std;
    BOOST_FOREACH( BndMapT::value_type& bndPair, bndNameBndSpecMap_){
      add_boundary_condition(bndPair.first, bcSpec);
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary( const std::string&      bndName,
                               Uintah::Patch::FaceType face,
                               const BndTypeEnum&      bndType,
                               const int               patchID,
                               const Uintah::BCGeomBase::ParticleBndSpec pBndSpec)
  {
    DBGBC << "adding boundary " << bndName << " of type " << bndType << " on patch " << patchID << std::endl;
    
    // if this boundary is a wall AND no particle boundaries have been specified, then default
    // the particle boundary to a fully elastic wall.
    Uintah::BCGeomBase::ParticleBndSpec myPBndSpec = pBndSpec;
    if (bndType == WALL && pBndSpec.bndType == Uintah::BCGeomBase::ParticleBndSpec::NOTSET) {
      myPBndSpec.bndType = Uintah::BCGeomBase::ParticleBndSpec::WALL;
      myPBndSpec.wallType = Uintah::BCGeomBase::ParticleBndSpec::ELASTIC;
      myPBndSpec.restitutionCoef = 1.0;
    }
    if ( bndNameBndSpecMap_.find(bndName) != bndNameBndSpecMap_.end() ) {
      DBGBC << " adding to existing \n";
      BndSpec& existingBndSpec = (*bndNameBndSpecMap_.find(bndName)).second;
      existingBndSpec.patchIDs.push_back(patchID);
    } else {
      DBGBC << " adding new \n";
      // this is the first time that we are adding this boundary. create the necessary info to store this
      BndSpec myBndSpec = {bndName, face, bndType, std::vector<int>(1, patchID), myPBndSpec };
      bndNameBndSpecMap_.insert( BndMapT::value_type(bndName, myBndSpec) );
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_auxiliary_boundary_condition( const std::string& srcVarName,
                                                   const std::string& newVarName,
                                                   const double newValue,
                                                   const BndCondTypeEnum newBCType )
  {
    BndCondSpec newBCSpec = {newVarName, "none", newValue, newBCType, DOUBLE_TYPE};
    add_auxiliary_boundary_condition(srcVarName, newBCSpec);
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_auxiliary_boundary_condition(const std::string& srcVarName,
                                                  BndCondSpec bcSpec)
  {
    BOOST_FOREACH( BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ ){
      BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = myBndSpec.find(srcVarName);
      if (myBndCondSpec) {
        add_boundary_condition(myBndSpec.name, bcSpec);
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary_mask( const BoundaryIterators& myIters,
                                    const std::string& bndName,
                                    const int& patchID )
  {
    using namespace std;
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      DBGBC << "BC " << bndName << " already exists in list of Iterators. Adding new iterator for " << bndName << " on patchID " << patchID << std::endl;
      (*bndNamePatchIDMaskMap_.find(bndName)).second.insert(pair<int, BoundaryIterators>(patchID, myIters));
    } else {
      DBGBC << "BC " << bndName << " does NOT Exist in list of Iterators. Adding new iterator for " << bndName << " on patchID " << patchID << std::endl;
      PatchIDBndItrMapT patchIDIterMap;
      patchIDIterMap.insert(pair<int, BoundaryIterators>(patchID, myIters));
      bndNamePatchIDMaskMap_.insert( pair< string, PatchIDBndItrMapT >(bndName, patchIDIterMap ) );
    }
  }

  //------------------------------------------------------------------------------------------------

  const std::vector<SpatialOps::IntVec>*
  BCHelper::get_edge_mask( const BndSpec& myBndSpec, const int& patchID ) const
  {
    const std::string bndName = myBndSpec.name;
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      const PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        return &(myIters.interiorEdgeCells);
      }
    }
    return NULL;
  }

  //------------------------------------------------------------------------------------------------
  
  const std::vector<int>*
  BCHelper::get_particles_bnd_mask( const BndSpec& myBndSpec,
                                    const int& patchID ) const
  {
    const std::string bndName = myBndSpec.name;
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      const PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        return myIters.particleIdx;
      }
    }
    return NULL;
  }

  //------------------------------------------------------------------------------------------------
  
  template<typename FieldT>
  const std::vector<SpatialOps::IntVec>*
  BCHelper::get_extra_bnd_mask( const BndSpec& myBndSpec,
                               const int& patchID ) const
  {
    const std::string bndName = myBndSpec.name;
    const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face);
    const bool isPlusSide = is_plus_side<FieldT>(myBndSpec.face);
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      const PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        if (isStagNorm && isPlusSide) {
          return &(myIters.extraPlusBndCells);
        } else {
          return &(myIters.extraBndCells);
        }
      }
    }
    return NULL;
  }
  
  //------------------------------------------------------------------------------------------------
  
  template<typename FieldT>
  const std::vector<SpatialOps::IntVec>*
  BCHelper::get_interior_bnd_mask( const BndSpec& myBndSpec,
                                  const int& patchID ) const
  {
    const std::string bndName = myBndSpec.name;
    const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face);
    const bool isPlusSide = is_plus_side<FieldT>(myBndSpec.face);
    
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      const PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        if (isStagNorm && isPlusSide) {
          return &(myIters.extraBndCells);
        } else {
          return &(myIters.interiorBndCells);
        }
      }
    }
    return NULL;
  }
  
  //------------------------------------------------------------------------------------------------
  
  Uintah::Iterator&
  BCHelper::get_uintah_extra_bnd_mask( const BndSpec& myBndSpec,
                                      const int& patchID )
  {
    const std::string bndName = myBndSpec.name;
    
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        return myIters.extraBndCellsUintah;
      }
    }
    
    std::ostringstream msg;
    msg << "ERROR: It looks like you were trying to grab a boundary iterator that doesn't exist! "
    << "This could be caused by requesting an iterator for a boundary/patch combination that is inconsistent with your input. "
    << "Otherwise, this is likely a major bug that needs to be addressed by a core Wasatch developer." << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  //------------------------------------------------------------------------------------------------
  
  template<typename FieldT>
  const SpatialOps::SpatialMask<FieldT>*
  BCHelper::get_spatial_mask( const BndSpec& myBndSpec,
                                  const int& patchID ) const
  {
    const std::string bndName = myBndSpec.name;
    
    if ( bndNamePatchIDMaskMap_.find(bndName) != bndNamePatchIDMaskMap_.end() ) {
      const PatchIDBndItrMapT& myMap = (*bndNamePatchIDMaskMap_.find(bndName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        return myIters.get_spatial_mask<FieldT>();
      }
    }
    
    std::ostringstream msg;
    msg << "ERROR: It looks like you were trying to grab a boundary iterator that doesn't exist! "
    << "This could be caused by requesting an iterator for a boundary/patch combination that is inconsistent with your input. "
    << "Otherwise, this is likely a major bug that needs to be addressed by a core Wasatch developer." << std::endl;
    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }


  //------------------------------------------------------------------------------------------------
  
  void BCHelper::print() const
  {
    BOOST_FOREACH( const BndMapT::value_type& bndNameBCSpecPair, bndNameBndSpecMap_ ){
      bndNameBCSpecPair.second.print();
    }
  }
  
  //------------------------------------------------------------------------------------------------

  void BCHelper::parse_boundary_conditions()
  {
    using namespace std;
    // loop over the material set
    BOOST_FOREACH( const Uintah::MaterialSubset* matSubSet, materials_->getVector() ) {
      
      // loop over materials
      for( int im=0; im<matSubSet->size(); ++im ) {
        
        const int materialID = matSubSet->get(im);
        
        // loop over local patches
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() ) {
          
          // loop over every patch in the patch subset
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
            
            const int patchID = patch->getID();
            DBGBC << "Patch ID = " << patchID << std::endl;
            
            std::vector<Uintah::Patch::FaceType> bndFaces;
            patch->getBoundaryFaces(bndFaces);
            
            // loop over the physical boundaries of this patch. These are the LOGICAL boundaries
            // and do NOT include intrusions
            BOOST_FOREACH(const Uintah::Patch::FaceType face, bndFaces) {
              
              // Get the number of "boundaries" (children) specified on this boundary face.
              // example: x- boundary face has a circle specified as inlet while the rest of the
              // face is specified as wall. This results in two "boundaries" or children.
              // the BCDataArray will store this list of children
              const Uintah::BCDataArray* bcDataArray = patch->getBCDataArray(face);
              
              // Grab the number of children on this boundary face
              const int numChildren = bcDataArray->getNumberChildren(materialID);
              
              DBGBC << "Face = " << face << std::endl;
              //bcDataArray->print();
              
              // now go over every child-boundary (sub-boundary) specified on this domain boundary face
              for( int chid = 0; chid<numChildren; ++chid ) {
                DBGBC << " child ID = " << chid << std::endl;
                
                // here is where the fun starts. Now we can get information about this boundary condition.
                // The BCDataArray stores information related to its children as BCGeomBase objects.
                // Each child is associated with a BCGeomBase object. Grab that
                Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(materialID,chid);
                const std::string bndName = thisGeom->getBCName();
                if (bndName.compare("NotSet")==0) {
                  std::ostringstream msg;
                  msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
                      << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
                  throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                }
                DBGBC << " boundary name = " << bndName << std::endl;
                DBGBC << " geom bndtype  = " << thisGeom->getBndType() << std::endl;
                BndTypeEnum bndType = select_bnd_type_enum(thisGeom->getBndType());
                add_boundary( bndName, face, bndType, patchID, thisGeom->getParticleBndSpec() );
                DBGBC << " boundary type = " << bndType << std::endl;
                
                //__________________________________________________________________________________
                Uintah::Iterator bndIter; // allocate iterator
                // get the iterator for the extracells for this child
                bcDataArray->getCellFaceIterator(materialID, bndIter, chid);
                
                BoundaryIterators myIters;
                DBGBC << " Size of uintah iterator for boundary: " << bndName << " = " << bndIter.size() << std::endl;
                pack_uintah_iterator_as_spatialops(face, patch, bndIter, myIters); // convert the Uintah iterator to a SpatialOps-friendly mask
                // store a pointer to the list of particle index that are near this boundary.
                myIters.particleIdx = Uintah::ParticlesHelper::get_boundary_particles(bndName,patchID);
                
                add_boundary_mask( myIters, bndName, patchID );
                
                //__________________________________________________________________________________
                // Now, each BCGeomObject has BCData associated with it. This BCData contains the list
                // of variables and types (Dirichlet, etc...), and values that the user specified
                // through the input file!
                Uintah::BCData bcData;
                thisGeom->getBCData(bcData);
                
                BOOST_FOREACH( Uintah::BoundCondBase* bndCondBase, bcData.getBCData() ) {
                  const std::string varName     = bndCondBase->getBCVariable();
                  const BndCondTypeEnum atomBCTypeEnum = select_bc_type_enum(bndCondBase->getBCType());
                  
                  DBGBC << " bc variable = " << varName << std::endl
                  << " bc type = "     << atomBCTypeEnum << std::endl;
                  
                  double doubleVal=0.0;
                  std::string functorName="none";
                  BCValueTypeEnum bcValType=INVALID_TYPE;
                  
                  switch ( bndCondBase->getValueType() ) {
                      
                    case Uintah::BoundCondBase::DOUBLE_TYPE: {
                      const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                      doubleVal = new_bc->getValue();
                      bcValType = DOUBLE_TYPE;
                      break;
                    }
                      
                    case Uintah::BoundCondBase::STRING_TYPE: {
                      const Uintah::BoundCond<std::string>* const new_bc = dynamic_cast<const Uintah::BoundCond<std::string>*>(bndCondBase);
                      functorName = new_bc->getValue();
                      bcValType = FUNCTOR_TYPE;
                      DBGBC << " functor name = " << functorName << std::endl;
                      break;
                    }
                    case Uintah::BoundCondBase::VECTOR_TYPE: {
                      // do nothing here... this is added for WARCHES support
                      break;
                    }
                    case Uintah::BoundCondBase::INT_TYPE: {
                      // do nothing here... this is added for RMCRT support
                      break;
                    }
                    default:
                    {
                      std::ostringstream msg;
                      msg << "ERROR: It looks like you have specified an unsupported datatype value for boundary " << bndName << ". "
                          << "Supported datatypes are: double, vector, and string (i.e. functor name)." << std::endl;
                      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                    }
                      break;
                  }
                  const BndCondSpec bndCondSpec = {varName, functorName, doubleVal, atomBCTypeEnum, bcValType};
                  add_boundary_condition(bndName, bndCondSpec);
                }
              } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
            } // boundary faces loop
            
            
            
            // INTERIOR BOUNDARY CONDITIONS
            if (patch->hasInteriorBoundaryFaces()) {
              
              for(Uintah::Patch::FaceType face_side = Uintah::Patch::startFace;
                  face_side <= Uintah::Patch::endFace; face_side=Uintah::Patch::nextFace(face_side))
              {
                
                // Get the number of "boundaries" (children) specified on this interior boundary face.
                // example: x- boundary face has a circle specified as inlet while the rest of the
                // face is specified as wall. This results in two "boundaries" or children.
                // the BCDataArray will store this list of children
                const Uintah::BCDataArray* bcDataArray = patch->getInteriorBndBCDataArray(face_side);
                
                // Grab the number of children on this boundary face
                const int numChildren = bcDataArray->getNumberChildren(materialID);
                                
                // now go over every child-boundary (sub-boundary) specified on this domain boundary face
                for( int chid = 0; chid<numChildren; ++chid ) {
                  DBGBC << " child ID = " << chid << std::endl;
                  
                  // here is where the fun starts. Now we can get information about this boundary condition.
                  // The BCDataArray stores information related to its children as BCGeomBase objects.
                  // Each child is associated with a BCGeomBase object. Grab that
                  Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(materialID,chid);
                  const std::string bndName = thisGeom->getBCName();
                  if (bndName.compare("NotSet")==0) {
                    std::ostringstream msg;
                    msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
                        << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
                    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                  }
                  DBGBC << " boundary name = " << bndName << std::endl;
                  DBGBC << " geom bndtype  = " << thisGeom->getBndType() << std::endl;
                  BndTypeEnum bndType = select_bnd_type_enum(thisGeom->getBndType());
                  add_boundary( bndName, face_side, bndType, patchID, thisGeom->getParticleBndSpec() );
                  DBGBC << " boundary type = " << bndType << std::endl;
                  
                  //__________________________________________________________________________________
                  Uintah::Iterator bndIter; // allocate iterator
                  // get the iterator for the extracells for this child
                  bcDataArray->getCellFaceIterator(materialID, bndIter, chid);
                  
                  BoundaryIterators myIters;
                  DBGBC << " Size of uintah iterator for boundary: " << bndName << " = " << bndIter.size() << std::endl;
                  pack_uintah_iterator_as_spatialops(face_side, patch, bndIter, myIters); // convert the Uintah iterator to a SpatialOps-friendly mask
                  // store a pointer to the list of particle index that are near this boundary.
                  myIters.particleIdx = Uintah::ParticlesHelper::get_boundary_particles(bndName,patchID);
                  
                  add_boundary_mask( myIters, bndName, patchID );
                  
                  //__________________________________________________________________________________
                  // Now, each BCGeomObject has BCData associated with it. This BCData contains the list
                  // of variables and types (Dirichlet, etc...), and values that the user specified
                  // through the input file!
                  Uintah::BCData bcData;
                  thisGeom->getBCData(bcData);
                  
                  BOOST_FOREACH( Uintah::BoundCondBase* bndCondBase, bcData.getBCData() ) {
                    const std::string varName     = bndCondBase->getBCVariable();
                    const BndCondTypeEnum atomBCTypeEnum = select_bc_type_enum(bndCondBase->getBCType());
                    
                    DBGBC << " bc variable = " << varName << std::endl
                          << " bc type = "     << atomBCTypeEnum << std::endl;
                    
                    double doubleVal=0.0;
                    std::string functorName="none";
                    BCValueTypeEnum bcValType=INVALID_TYPE;
                    
                    switch ( bndCondBase->getValueType() ) {
                        
                      case Uintah::BoundCondBase::DOUBLE_TYPE: {
                        const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                        doubleVal = new_bc->getValue();
                        bcValType = DOUBLE_TYPE;
                        break;
                      }
                        
                      case Uintah::BoundCondBase::STRING_TYPE: {
                        const Uintah::BoundCond<std::string>* const new_bc = dynamic_cast<const Uintah::BoundCond<std::string>*>(bndCondBase);
                        functorName = new_bc->getValue();
                        bcValType = FUNCTOR_TYPE;
                        DBGBC << " functor name = " << functorName << std::endl;
                        break;
                      }
                      case Uintah::BoundCondBase::VECTOR_TYPE: {
                        // do nothing here... this is added for WARCHES support
                        break;
                      }
                      case Uintah::BoundCondBase::INT_TYPE: {
                        // do nothing here... this is added for RMCRT support
                        break;
                      }
                      default:
                      {
                        std::ostringstream msg;
                        msg << "ERROR: It looks like you have specified an unsupported datatype value for boundary " << bndName << ". "
                            << "Supported datatypes are: double, vector, and string (i.e. functor name)." << std::endl;
                        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                      }
                        break;
                    }
                    const BndCondSpec bndCondSpec = {varName, functorName, doubleVal, atomBCTypeEnum, bcValType};
                    add_boundary_condition(bndName, bndCondSpec);
                  }
                } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
              }
            }
          } // patch loop
        } // patch subset loop
      } // material loop
    } // material subset loop
  }

  //------------------------------------------------------------------------------------------------
  
  const BndMapT& BCHelper::get_boundary_information() const
  {
    return bndNameBndSpecMap_;
  }

  //------------------------------------------------------------------------------------------------
  
  bool BCHelper::has_boundaries() const
  {
    return !bndNameBndSpecMap_.empty();
  }

  //------------------------------------------------------------------------------------------------
  template<typename SrcT, typename TargetT>
  void BCHelper::create_dummy_dependency( const Expr::Tag& attachDepToThisTag,
                                          const Expr::TagList dependencies,
                                          const Category taskCat)
  {
    Expr::ExpressionFactory& factory = *(grafCat_[taskCat]->exprFactory);
    std::string phiName = attachDepToThisTag.name();
    
    // check if we already have an entry for phiname
    BCFunctorMap::iterator iter = bcFunctorMap_.find(phiName);
    size_t nMods = 0;

    Expr::Tag dummyTag(phiName + "_dummy_dependency", Expr::STATE_NONE );
    
    if ( iter != bcFunctorMap_.end() ) {
      (*iter).second.insert(dummyTag.name());
    } else {
      BCFunctorMap::mapped_type functorSet;
      nMods = (*iter).second.size();
      std::ostringstream msg;
      msg << nMods;
      dummyTag.name() = phiName + "_dummy_dependency_" + msg.str();
      functorSet.insert( dummyTag.name() );
      bcFunctorMap_.insert( BCFunctorMap::value_type(phiName,functorSet) );
    }

    // if the dependency was already added then return
    if (factory.have_entry(dummyTag)) {
      return;
    }
    
    // register the null dependency
    typedef typename NullExpression<SrcT, TargetT>::Builder NullExpr;
    factory.register_expression(scinew NullExpr(dummyTag, dependencies), true);
  }
  
  //------------------------------------------------------------------------------------------------
  
  template <typename FieldT>
  void BCHelper::apply_boundary_condition( const Expr::Tag& varTag,
                                           const Category& taskCat,
                                           const bool setOnExtraOnly )

  {            
    using namespace std;
    string fieldName = varTag.name();
    Expr::ExpressionFactory& factory = *(grafCat_[taskCat]->exprFactory);
    
    //_____________________________________________________________________________________
    // process functors... this is an ugly part of the code but we
    // have to deal with this because Uintah doesn't allow a given task
    // to run on different patches with different requires. We also can't
    // get a bc graph to play nicely with the time advance graphs.
    // this block will essentially enforce the same dependencies across
    // all patches by adding functor expressions on them. those patches
    // that do NOT have that functor associated with any of their boundaries
    // will just expose the dependencies advertised by the functor but will
    // accomplish nothing else because that functor doesn't have any bc points
    // associated with it.    
    BOOST_FOREACH( const Uintah::MaterialSubset* matSubSet, materials_->getVector() )
    {
      BOOST_FOREACH( const int im, matSubSet->getVector() )
      {
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() )
        {
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() )
          {
            const int patchID = patch->getID();
            BCFunctorMap::iterator iter = bcFunctorMap_.begin();
            while ( iter != bcFunctorMap_.end() ) {
              string functorPhiName = (*iter).first;
              if ( functorPhiName.compare(fieldName) == 0 ) {
                // get the functor set associated with this field
                BCFunctorMap::mapped_type::const_iterator functorIter = (*iter).second.begin();
                while( functorIter != (*iter).second.end() ){
                  const string& functorName = *functorIter;
                  const Expr::Tag modTag = Expr::Tag(functorName,Expr::STATE_NONE);
                  if (factory.have_entry(modTag)) {
                    DBGBC << "attaching dummy modifier " << functorName << " on field " << varTag << " on patch " << patchID << endl;
                    factory.attach_modifier_expression( modTag, varTag, patchID, true );
                  }
                  ++functorIter;
                } // while
              } // if
              ++iter;
            } // while bcFunctorMap_
          } // patches
        } // localPatches_
      } // matSubSet
    } // materials_
    
    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(fieldName);
      
      if (myBndCondSpec) {        
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() )
        {
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() )
          {
            const int patchID = patch->getID();
            //_____________________________________________________________________________________
            // check if we have this patchID in the list of patchIDs
            if( myBndSpec.has_patch(patchID) ){
              //____________________________________________________________________________________
              // get the patch info from which we can get the operators database
              const PatchInfoMap::const_iterator ipi = patchInfoMap_.find( patchID );
              assert( ipi != patchInfoMap_.end() );
              const SpatialOps::OperatorDatabase& opdb = *(ipi->second.operators);

              //____________________________________________________________________________________
              // create unique names for the modifier expressions
              const string strPatchID = number_to_string(patchID);
              
              Expr::Tag modTag;
              
              // create bc expressions. These are not created from the input file.
              if( myBndCondSpec->is_functor() ) { // functor bc
                modTag = Expr::Tag( myBndCondSpec->functorName, Expr::STATE_NONE );
              }
              else{ // constant bc
                modTag = Expr::Tag( fieldName + "_" + Expr::context2str(varTag.context()) + "_bc_" + myBndSpec.name + "_patch_" + strPatchID, Expr::STATE_NONE );
                factory.register_expression( new typename ConstantBC<FieldT>::Builder( modTag, myBndCondSpec->value ), true );
              }
              
              // attach the modifier expression to the target expression
              factory.attach_modifier_expression( modTag, varTag, patchID, true );
              
              // now retrieve the modifier expression and set the ghost and interior points
              BoundaryConditionBase<FieldT>& modExpr =
              dynamic_cast<BoundaryConditionBase<FieldT>&>( factory.retrieve_modifier_expression( modTag, patchID, false ) );
              
              // the following is needed for bc expressions that require global uintah indexing, e.g. TurbulentInletBC
              const SCIRun::IntVector sciPatchCellOffset = patch->getExtraCellLowIndex(1);
              const IntVecT patchCellOffset(sciPatchCellOffset.x(), sciPatchCellOffset.y(), sciPatchCellOffset.z());
              modExpr.set_patch_cell_offset(patchCellOffset);
              
              // check if this is a staggered field on a face in the same staggered direction (XVolField on x- Face)
              // and whether this a Neumann bc or not.
              const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face);// && myBndCondSpec->bcType!=NEUMANN;
              modExpr.set_staggered_normal(isStagNorm);
              
              // set the ghost and interior points as well as coefficients
              double ci, cg;
              get_bcop_coefs<FieldT>(opdb, myBndSpec, *myBndCondSpec, ci, cg, setOnExtraOnly);

              // get the unit normal
              const Uintah::IntVector uNorm = patch->getFaceDirection(myBndSpec.face);
              const IntVecT unitNormal( uNorm.x(), uNorm.y(), uNorm.z() );
              modExpr.set_boundary_normal(unitNormal);
              modExpr.set_bc_type(myBndCondSpec->bcType);
              modExpr.set_face_type(myBndSpec.face);
              modExpr.set_extra_only(setOnExtraOnly);
              modExpr.set_ghost_coef( cg );
              modExpr.set_ghost_points( get_extra_bnd_mask<FieldT>(myBndSpec, patchID) );
              
              modExpr.set_interior_coef( ci );
              modExpr.set_interior_points( get_interior_bnd_mask<FieldT>(myBndSpec,patchID) );
              
              modExpr.set_boundary_particles(get_particles_bnd_mask(myBndSpec,patchID));
              // tsaad: do not delete this. this could be needed for some outflow/open boundary conditions
              //modExpr.set_interior_edge_points( get_edge_mask(myBndSpec,patchID) );
              modExpr.set_spatial_mask(get_spatial_mask<FieldT>(myBndSpec,patchID));
              modExpr.set_svol_spatial_mask(get_spatial_mask<SVolField>(myBndSpec,patchID));
            }
          }
        }
      }
    }
  }
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::synchronize_pressure_expression()
  {
    Expr::ExpressionFactory& factory = *(grafCat_[ADVANCE_SOLUTION]->exprFactory);
    if (!factory.have_entry( pressure_tag() )) return;
    BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() ) {
      // loop over every patch in the patch subset
      BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
        const int patchID = patch->getID();
        // go through the patch ids and pass the bchelper to the pressure expression
        Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( pressure_tag(), patchID, true ) );
        pexpr.set_bchelper(this); // add the bchelper class to the pressure expression on ALL patches
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::update_pressure_matrix( Uintah::CCVariable<Uintah::Stencil4>& pMatrix,
                                         const SVolField* const volFrac,
                                         const Uintah::Patch* patch )
  {
    const int patchID = patch->getID();
    const Uintah::Vector spacing = patch->dCell();
    const double dx = spacing[0];
    const double dy = spacing[1];
    const double dz = spacing[2];
    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    const double dz2 = dz*dz;
    
    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second; // get the boundary specification
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pressure_tag().name()); // get the bc spec - we will check if the user specified anything for pressure here
      const Uintah::IntVector unitNormal = patch->getFaceDirection(myBndSpec.face);
      //_____________________________________________________________________________________
      // check if we have this patchID in the list of patchIDs
      // here are the scenarios here:
      /*
       1. Inlet/Wall/Moving wall: dp/dx = 0 -> p_outside = p_inside. Therefore for d2p/dx2 = (p_{-1} - 2 p_0 + p_1)/dx2, p_1 = p_0, therefore we decrement the coefficient for p0 by 1.
       2. OUTFLOW/OPEN: p_outside = - p_inside -> we augment the coefficient for p_0
       3. Intrusion: do NOT modify the coefficient matrix since it will be modified inside the pressure expression when modifying the matrix for intrusions
       */
      if( myBndSpec.has_patch(patchID) ){
        Uintah::Iterator& bndMask = get_uintah_extra_bnd_mask(myBndSpec,patchID);
        
        double sign = (myBndSpec.type == OUTFLOW || myBndSpec.type == OPEN) ? 1.0 : -1.0; // For OUTFLOW/OPEN boundaries, augment the P0
        if (myBndCondSpec) {
          if (myBndCondSpec->bcType == DIRICHLET) { // DIRICHLET on pressure
            sign = 1.0;
          }
        }
        
        for( bndMask.reset(); !bndMask.done(); ++bndMask ){
          Uintah::Stencil4& coefs = pMatrix[*bndMask - unitNormal];
          
          // if we are inside a solid, then don't do anything because we already handle this in the pressure expression
          if( volFrac ){
            const Uintah::IntVector iCell = *bndMask - unitNormal - patch->getExtraCellLowIndex(1);
            const SpatialOps::IntVec iiCell(iCell.x(), iCell.y(), iCell.z() );
            if ((*volFrac)(iiCell) < 1.0)
              continue;
          }
          
          //
          switch(myBndSpec.face){
            case Uintah::Patch::xminus: coefs.w = 0.0; coefs.p += sign/dx2; break;
            case Uintah::Patch::xplus :                coefs.p += sign/dx2; break;
            case Uintah::Patch::yminus: coefs.s = 0.0; coefs.p += sign/dy2; break;
            case Uintah::Patch::yplus :                coefs.p += sign/dy2; break;
            case Uintah::Patch::zminus: coefs.b = 0.0; coefs.p += sign/dz2; break;
            case Uintah::Patch::zplus :                coefs.p += sign/dz2; break;
            default:                                                        break;
          }
        }
      }
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::apply_pressure_bc( SVolField& pressureField,
                                    const Uintah::Patch* patch )
  {
    typedef std::vector<SpatialOps::IntVec> MaskT;
    
    const int patchID = patch->getID();

    const Uintah::Vector res = patch->dCell();
    const double dx = res[0];
    const double dy = res[1];
    const double dz = res[2];

    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pressure_tag().name());
      
      double spacing = 1.0;
      switch( myBndSpec.face ){
        case Uintah::Patch::xminus: case Uintah::Patch::xplus :  spacing = dx;  break;
        case Uintah::Patch::yminus: case Uintah::Patch::yplus :  spacing = dy;  break;
        case Uintah::Patch::zminus: case Uintah::Patch::zplus :  spacing = dz;  break;
        default:                                                                break;
      } // switch
      
      if (myBndCondSpec) {
        //_____________________________________________________________________________________
        // check if we have this patchID in the list of patchIDs
        if (myBndSpec.has_patch(patchID))
        {
          const MaskT* iBndMask = get_interior_bnd_mask<SVolField>(myBndSpec,patchID);
          const MaskT* eBndMask = get_extra_bnd_mask<SVolField>(myBndSpec,patchID);
          MaskT::const_iterator ii = iBndMask->begin();
          MaskT::const_iterator ig = eBndMask->begin();
          if(!iBndMask || !eBndMask) return;
          if (myBndSpec.type == OUTFLOW || myBndSpec.type == OPEN) {
            for (; ii != iBndMask->end(); ++ii, ++ig) {
              pressureField(*ig) = -pressureField(*ii);
            }
          } else {
            switch (myBndCondSpec->bcType) {
              case DIRICHLET:
                for (; ii != iBndMask->end(); ++ii, ++ig) {
                  pressureField(*ig) = 2.0*myBndCondSpec->value - pressureField(*ii);
                }
                break;
              case NEUMANN:
                for (; ii != iBndMask->end(); ++ii, ++ig) {
                  pressureField(*ig) = spacing * myBndCondSpec->value + pressureField(*ii);
                }
                break;
              default:
                break;
            }
          }
        }
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::update_pressure_rhs( SVolField& pressureRHS,
                                      const Uintah::Patch* patch )
  {
    typedef std::vector<SpatialOps::IntVec> MaskT;
    
    const int patchID = patch->getID();
    
    const Uintah::Vector res = patch->dCell();
    const double dx = res[0];
    const double dy = res[1];
    const double dz = res[2];
    
    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pressure_tag().name());
      
      double spacing = 1.0;
      switch( myBndSpec.face ){
        case Uintah::Patch::xminus: case Uintah::Patch::xplus :  spacing = dx;  break;
        case Uintah::Patch::yminus: case Uintah::Patch::yplus :  spacing = dy;  break;
        case Uintah::Patch::zminus: case Uintah::Patch::zplus :  spacing = dz;  break;
        default:                                                                break;
      } // switch
      
      if (myBndCondSpec) {
        //_____________________________________________________________________________________
        // check if we have this patchID in the list of patchIDs
        if (myBndSpec.has_patch(patchID))
        {
          const MaskT* iBndMask = get_interior_bnd_mask<SVolField>(myBndSpec,patchID);
          const MaskT* eBndMask = get_extra_bnd_mask<SVolField>(myBndSpec,patchID);
          MaskT::const_iterator ii = iBndMask->begin();
          MaskT::const_iterator ig = eBndMask->begin();
          
          if(!iBndMask || !eBndMask) return;
          
          if (myBndSpec.type == OUTFLOW || myBndSpec.type == OPEN) {
            // do nothing for now
          } else {
            switch (myBndCondSpec->bcType) {
              case DIRICHLET:
                for (; ii != iBndMask->end(); ++ii, ++ig) {
                  pressureRHS(*ig) = 2.0*myBndCondSpec->value/spacing/spacing;
                }
                break;
              case NEUMANN:
                for (; ii != iBndMask->end(); ++ii, ++ig) {
                  pressureRHS(*ig) =  myBndCondSpec->value/spacing;
                }
                break;
              default:
                break;
            }
          }
        }
      }
    }
  }

  //------------------------------------------------------------------------------------------------

  //==========================================================================
  // Explicit template instantiation for supported versions of this class
  #include <spatialops/structured/FVStaggered.h>

#define INSTANTIATE_BC_TYPES(VOLT) \
  template void BCHelper::apply_boundary_condition< VOLT >( const Expr::Tag& varTag,            \
                                                            const Category& taskCat,            \
                                                            const bool setOnExtraOnly);         \
  template void BCHelper::create_dummy_dependency< VOLT, SpatialOps::SVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void BCHelper::create_dummy_dependency< VOLT, SpatialOps::XVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void BCHelper::create_dummy_dependency< VOLT, SpatialOps::YVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void BCHelper::create_dummy_dependency< VOLT, SpatialOps::ZVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void BCHelper::create_dummy_dependency< SpatialOps::SingleValueField, VOLT >( const Expr::Tag& attachDepToThisTag, \
                                                                                 const Expr::TagList dependencies,     \
                                                                                 const Category taskCat );

  INSTANTIATE_BC_TYPES(SpatialOps::SVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::XVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::YVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::ZVolField);
  
  template void BCHelper::apply_boundary_condition< ParticleField >( const Expr::Tag& varTag,
                                                                     const Category& taskCat,
                                                                     const bool setOnExtraOnly);

} // class BCHelper
