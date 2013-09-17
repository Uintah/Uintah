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

#include "BCHelper.h"

//-- C++ Includes --//
#include <vector>
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

//-- ExprLib Includes --//
#include <expression/ExprLib.h>
#include <expression/ExpressionFactory.h>

//-- Wasatch Includes --//
#include "FieldTypes.h"
#include "ParseTools.h"
#include "Expressions/BoundaryConditions/ConstantBC.h"
#include "Expressions/BoundaryConditions/ParabolicBC.h"
#include "Expressions/BoundaryConditions/BoundaryConditionBase.h"

namespace Wasatch {
  
  BndCondTypeEnum select_bc_type_enum( const std::string& bcTypeStr )
  {
    if      ( bcTypeStr == "Dirichlet" )     return DIRICHLET;
    else if ( bcTypeStr == "Neumann" )       return NEUMANN;
    else                                     return UNSUPPORTED;
  }

  BndTypeEnum select_bnd_type_enum( const std::string& bcTypeStr )
  {
    if      ( bcTypeStr == "Wall" )          return WALL;
    else if ( bcTypeStr == "VelocityInlet" ) return VELOCITYINLET;
    else if ( bcTypeStr == "Pressure" )      return PRESSURE;
    else if ( bcTypeStr == "Outflow" )       return OUTFLOW;
    else if ( bcTypeStr == "None" )          return USER;
    else                                     return INVALID;
  }

  const std::string bc_type_enum_to_string( const BndCondTypeEnum bcTypeEnum )
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

  const std::string bnd_type_enum_to_string( const BndTypeEnum bcTypeEnum )
  {
    switch (bcTypeEnum) {
      case WALL:
        return "Wall";
        break;
      case VELOCITYINLET:
        return "VelocityInlet";
        break;
      case PRESSURE:
        return "Pressure";
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
  OST& operator<<( OST& os, const BndTypeEnum bcTypeEnum )
  {
    os << bnd_type_enum_to_string(bcTypeEnum);
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
            break;
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
            break;
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
        break;
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
                                           BoundaryIterators& myBndIters)
  {
    namespace SS = SpatialOps::structured;
        
    std::vector<SS::IntVec>& extraBndSOIter    = myBndIters.extraBndCells;
    std::vector<SS::IntVec>& intBndSOIter      = myBndIters.interiorBndCells;
    std::vector<SS::IntVec>& extraPlusBndCells = myBndIters.extraPlusBndCells;
    
    // MAJOR WARNING HERE - WHEN WE MOVE TO RUNTIME GHOST CELLS, WE NEED TO USE THE APPROPRIATE PATCH OFFSET
    const Uintah::IntVector patchCellOffset = patch->getExtraCellLowIndex(1);
    Uintah::IntVector unitNormal = patch->faceDirection(face); // this is needed to construct interior cells
    Uintah::IntVector bcPointIJK;

    for( bndIter.reset(); !bndIter.done(); bndIter++ )
    {
      bcPointIJK = *bndIter - patchCellOffset;
      extraBndSOIter.push_back(SS::IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
      bcPointIJK -= unitNormal;
      intBndSOIter.push_back(SS::IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
    }
    
    // if we are on a plus face, we will most likely need a plus-face iterator for staggered fields
    if (face == Uintah::Patch::xplus || face == Uintah::Patch::yplus || face == Uintah::Patch::zplus )
    {
      for( bndIter.reset(); !bndIter.done(); bndIter++ ) {
        bcPointIJK = *bndIter - patchCellOffset + unitNormal;
        extraPlusBndCells.push_back(SS::IntVec(bcPointIJK.x(), bcPointIJK.y(), bcPointIJK.z()));
      }
    }
  }

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
                      const BCFunctorMap& bcFunctorMap )
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
  {}    
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary_condition( const std::string&     bcName,
                                         const BndCondSpec& bcSpec )
  {
    using namespace std;
    if ( bndNameBndSpecMap_.find(bcName) != bndNameBndSpecMap_.end() ) {
      BndSpec& existingBCSpec = (*bndNameBndSpecMap_.find(bcName)).second;
      vector<BndCondSpec>& bcSpecVec = existingBCSpec.bcSpecVec;
      vector<BndCondSpec>::iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), bcSpec);
      if ( it == bcSpecVec.end() ) {
        DBGBC << "adding bc " << bcSpec.varName << " on " << bcName << " \n";
        bcSpecVec.push_back(bcSpec);
      } else {
        DBGBC << "bc " << bcSpec.varName << " already exists on " << bcName << ". skipping \n";
      }
    } else {
      DBGBC << " ERROR! boundary face " << bcName << " does not exist!!! \n";
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_physical_boundary_type( const std::string&      bcfaceName,
                                             Uintah::Patch::FaceType face,
                                             const BndTypeEnum& physicalBCType,
                                             const int               patchID )
  {
    DBGBC << "adding physical boundary " << bcfaceName << " of type " << physicalBCType << " on patch " << patchID << std::endl;
    if ( bndNameBndSpecMap_.find(bcfaceName) != bndNameBndSpecMap_.end() ) {
      DBGBC << " adding to existing \n";
      BndSpec& existingBCSpec = (*bndNameBndSpecMap_.find(bcfaceName)).second;
      existingBCSpec.patchIDs.push_back(patchID);
    } else {
      DBGBC << " adding new \n";
      // if this is the first time that we are adding this variable, then create the necessary info to store this
      BndSpec myPhysBCSpec = {bcfaceName, face, physicalBCType, std::vector<int>(1, patchID) };
      bndNameBndSpecMap_.insert( BndMapT::value_type(bcfaceName, myPhysBCSpec) );
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_auxiliary_boundary_condition( const std::string& srcVarName,
                                                   const std::string& newVarName,
                                                   const double& newValue,
                                                   const BndCondTypeEnum newBCType )
  {
    namespace SS = SpatialOps::structured;
    BndCondSpec newBCSpec = {newVarName, "none", newValue,SS::Numeric3Vec<double>(0,0,0), newBCType, DOUBLE_TYPE};
    add_auxiliary_boundary_condition(srcVarName, newBCSpec);
  }
  
  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_auxiliary_boundary_condition( const std::string& srcVarName,
                                                   const std::string& newVarName,
                                                   const std::string& functorName,
                                                   const BndCondTypeEnum newBCType )
  {
    namespace SS = SpatialOps::structured;
    BndCondSpec newBCSpec = {newVarName, functorName, 0.0, SS::Numeric3Vec<double>(0,0,0), newBCType, FUNCTOR_TYPE};
    add_auxiliary_boundary_condition(srcVarName, newBCSpec);
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_auxiliary_boundary_condition(const std::string& srcVarName,
                                                  BndCondSpec bcSpec)
  {
    using namespace std;
    BOOST_FOREACH( BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = myBndSpec.find(srcVarName);
      if (myBndCondSpec) {
        add_boundary_condition(myBndSpec.bndName, bcSpec);
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::add_boundary_iterator( const BoundaryIterators& myIters,
                                        const std::string& bcName,
                                        const int& patchID )
  {
    using namespace std;
    if ( bcNamePatchIDMaskMap_.find(bcName) != bcNamePatchIDMaskMap_.end() ) {
      DBGBC << "BC " << bcName << " already exists in list of Iterators. Adding new iterator for " << bcName << " on patchID " << patchID << std::endl;
      (*bcNamePatchIDMaskMap_.find(bcName)).second.insert(pair<int, BoundaryIterators>(patchID, myIters));
    } else {
      DBGBC << "BC " << bcName << " does NOT Exist in list of Iterators. Adding new iterator for " << bcName << " on patchID " << patchID << std::endl;
      patchIDBndItrMapT patchIDIterMap;
      patchIDIterMap.insert(pair<int, BoundaryIterators>(patchID, myIters));
      bcNamePatchIDMaskMap_.insert( pair< string, patchIDBndItrMapT >(bcName, patchIDIterMap ) );
    }
  }

  //------------------------------------------------------------------------------------------------
  
  template<typename FieldT>
  const std::vector<SpatialOps::structured::IntVec>*
  BCHelper::get_extra_bnd_mask( const BndSpec& myBndSpec,
                               const int& patchID ) const
  {
    const std::string bcName = myBndSpec.bndName;
    const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face);
    const bool isPlusSide = is_plus_side<FieldT>(myBndSpec.face);
    if ( bcNamePatchIDMaskMap_.find(bcName) != bcNamePatchIDMaskMap_.end() ) {
      const patchIDBndItrMapT& myMap = (*bcNamePatchIDMaskMap_.find(bcName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        if (isStagNorm && isPlusSide) {
          return &(myIters.extraPlusBndCells);
        } else {
          return &(myIters.extraBndCells);
        }
      }
    } else {
      return NULL;
    }
  }

  //------------------------------------------------------------------------------------------------
  
  template<typename FieldT>
  const std::vector<SpatialOps::structured::IntVec>*
  BCHelper::get_interior_bnd_mask( const BndSpec& myBndSpec,
                                  const int& patchID ) const
  {
    const std::string bcName = myBndSpec.bndName;
    const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face);
    const bool isPlusSide = is_plus_side<FieldT>(myBndSpec.face);
    
    if ( bcNamePatchIDMaskMap_.find(bcName) != bcNamePatchIDMaskMap_.end() ) {
      const patchIDBndItrMapT& myMap = (*bcNamePatchIDMaskMap_.find(bcName)).second;
      if ( myMap.find(patchID) != myMap.end() ) {
        const BoundaryIterators& myIters = (*myMap.find(patchID)).second;
        if (isStagNorm && isPlusSide) {
          return &(myIters.extraBndCells);
        } else {
          return &(myIters.interiorBndCells);
        }
      }
    } else {
      return NULL;
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void BCHelper::print() const
  {
    BOOST_FOREACH( const BndMapT::value_type& bcNameBCSpecPair, bndNameBndSpecMap_ ){
      bcNameBCSpecPair.second.print();
    }
  }
  
  //------------------------------------------------------------------------------------------------

  void BCHelper::parse_boundary_conditions()
  {
    namespace SS = SpatialOps::structured;
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
            std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();
            
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
              
              // now go over every boundary or child specified on this boundary face
              for( int chid = 0; chid<numChildren; ++chid ) {
                DBGBC << " child ID = " << chid << std::endl;

                // here is where the fun starts. Now we can get information about this boundary condition.
                // The BCDataArray stores information related to its children as BCGeomBase objects.
                // Each child is associated with a BCGeomBase object. Grab that
                Uintah::BCGeomBase* thisGeom = bcDataArray->getChild(materialID,chid);
                const std::string bcName = thisGeom->getBCName();
                if (bcName.compare("NotSet")==0) {
                  std::ostringstream msg;
                  msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
                      << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
                  throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                }
                DBGBC << " bc name = " << bcName << std::endl;                

                BndTypeEnum physicalBCType = select_bnd_type_enum(thisGeom->getBCType());
                add_physical_boundary_type( bcName, face, physicalBCType, patchID);
                DBGBC << " bc name = " << bcName << std::endl
                          << " physical bc type = " << physicalBCType << std::endl;
                
                //__________________________________________________________________________________
                Uintah::Iterator bndIter; // allocate iterator
                // get the iterator for the extracells for this child
                bcDataArray->getCellFaceIterator(materialID, bndIter, chid);
                
                BoundaryIterators myIters;
                pack_uintah_iterator_as_spatialops(face, patch, bndIter, myIters); // convert the Uintah iterator to a SpatialOps-friendly mask
                add_boundary_iterator( myIters, bcName, patchID );
                
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
                  
                  double doubleVal=0;
                  SpatialOps::structured::Numeric3Vec<double> vecVal(0.0, 0.0, 0.0);
                  std::string functorName="none";
                  BCValueTypeEnum bcValType=INVALID_TYPE;
                  
                  switch ( bndCondBase->getValueType() ) {
                      
                    case Uintah::BoundCondBase::DOUBLE_TYPE: {
                      const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                      doubleVal = new_bc->getValue();
                      bcValType = DOUBLE_TYPE;
                      break;
                    }
                      
                    case Uintah::BoundCondBase::VECTOR_TYPE: {
                      const Uintah::BoundCond<Uintah::Vector>* const new_bc = dynamic_cast<const Uintah::BoundCond<Uintah::Vector>*>(bndCondBase);
                      Uintah::Vector uintahVecVal = new_bc->getValue();
                      vecVal[0] = uintahVecVal.x();
                      vecVal[1] = uintahVecVal.y();
                      vecVal[2] = uintahVecVal.z();
                      bcValType = VECTOR_TYPE;
                      break;
                    }
                      
                    case Uintah::BoundCondBase::STRING_TYPE: {
                      const Uintah::BoundCond<std::string>* const new_bc = dynamic_cast<const Uintah::BoundCond<std::string>*>(bndCondBase);
                      functorName = new_bc->getValue();
                      bcValType = FUNCTOR_TYPE;
                      DBGBC << " functor name = " << functorName << std::endl;                      
                      break;
                    }
                      
                    default:
                    {
                      std::ostringstream msg;
                      msg << "ERROR: It looks like you have specified an unsupported datatype value for boundary " << bcName << ". "
                      << "Supported datatypes are: double, vector, and string (i.e. functor name)." << std::endl;
                      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                    }
                      break;
                  }
                  
                  //const BoundarySpec thisBndSpec = {face, patch->getID(), bcName, varName, functorName, doubleVal, vecVal, atomBCTypeEnum, USER, bcValType};
                  //add_boundary_condition(thisBndSpec);
                  
                  // NEW STUFF
                  const BndCondSpec bndCondSpecNew = {varName, functorName, doubleVal, vecVal, atomBCTypeEnum, bcValType};
                  add_boundary_condition(bcName, bndCondSpecNew);
                }
              } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
            } // boundary faces loop
          } // patch loop
        } // patch subset loop
      } // material loop
    } // material subset loop
    print();
  }
  
  //------------------------------------------------------------------------------------------------

  template <typename FieldT>
  void BCHelper::apply_boundary_condition( const Expr::Tag& varTag,
                                           const Category& taskCat,
                                           const bool setOnExtraOnly )

  {            
    using namespace std;
    std::string fieldName = varTag.name();

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
    BOOST_FOREACH( const Uintah::MaterialSubset* matSubSet, materials_->getVector() ) {
      BOOST_FOREACH( const int im, matSubSet->getVector() ) {
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() ) {
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
            const int patchID = patch->getID();
            BCFunctorMap::const_iterator iter = bcFunctorMap_.begin();
            while ( iter != bcFunctorMap_.end() ) {
              string functorPhiName = (*iter).first;
              if ( functorPhiName.compare(fieldName) == 0 ) {
                // get the functor set associated with this field
                BCFunctorMap::mapped_type::const_iterator functorIter = (*iter).second.begin();
                while( functorIter != (*iter).second.end() ){
                  DBGBC << "attaching dummy modifier..." << endl;
                  const string& functorName = *functorIter;
                  const Expr::Tag modTag = Expr::Tag(functorName,Expr::STATE_NONE);
                  factory.attach_modifier_expression( modTag, varTag, patchID, true );
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
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() ) {
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
            const int patchID = patch->getID();

            //_____________________________________________________________________________________
            // get the patch info from which we can get the operators database
            const PatchInfoMap::const_iterator ipi = patchInfoMap_.find( patchID );
            assert( ipi != patchInfoMap_.end() );
            const SpatialOps::OperatorDatabase& opdb = *(ipi->second.operators);
            
            //_____________________________________________________________________________________
            // check if we have this patchID in the list of patchIDs
            if (myBndSpec.find_patchid(patchID)) {              
              // create unique names for the modifier expressions
              const string strPatchID = number_to_string(patchID);
              
              Expr::Tag modTag;
              Expr::ExpressionBuilder* builder = NULL;
              
              // create constant bc expressions. These are not created from the input file.
              if( myBndCondSpec->bcValType != FUNCTOR_TYPE ){ // constant bc
                modTag = Expr::Tag( fieldName + "_bc_" + myBndSpec.bndName + "_patch_" + strPatchID, Expr::STATE_NONE );
                builder = new typename ConstantBC<FieldT>::Builder( modTag, myBndCondSpec->value );
                factory.register_expression( builder, true );
              }
              else{ // expression bc
                modTag = Expr::Tag( myBndCondSpec->functorName, Expr::STATE_NONE );
              }
              
              // attach the modifier expression to the target expression
              factory.attach_modifier_expression( modTag, varTag, patchID, true );
              
              // now retrieve the modifier expression and set the ghost and interior points
              BoundaryConditionBase<FieldT>& modExpr =
              dynamic_cast<BoundaryConditionBase<FieldT>&>( factory.retrieve_modifier_expression( modTag, patchID, false ) );
              
              // this is needed for bc expressions that require global uintah indexing, e.g. TurbulentInletBC
              const SCIRun::IntVector sciPatchCellOffset = patch->getExtraCellLowIndex(1);
              const IntVecT patchCellOffset(sciPatchCellOffset.x(), sciPatchCellOffset.y(), sciPatchCellOffset.z());
              modExpr.set_patch_cell_offset(patchCellOffset);
              
              // check if this is a staggered field on a face in the same staggered direction (XVolField on x- Face)
              // and whether this a Neumann bc or not.
              const bool isStagNorm = is_staggered_normal<FieldT>(myBndSpec.face) && myBndCondSpec->bcType!=NEUMANN;
              modExpr.set_staggered(isStagNorm);
              
              // set the ghost and interior points as well as coefficients
              double ci, cg;
              get_bcop_coefs<FieldT>(opdb, myBndSpec, *myBndCondSpec, ci, cg, setOnExtraOnly);

              modExpr.set_ghost_coef( cg );
              modExpr.set_ghost_points( get_extra_bnd_mask<FieldT>(myBndSpec, patchID) );
              modExpr.set_interior_coef( ci );
              modExpr.set_interior_points( get_interior_bnd_mask<FieldT>(myBndSpec,patchID) );              
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
  template void BCHelper::apply_boundary_condition< VOLT >( const Expr::Tag& varTag, \
                                                            const Category& taskCat, \
                                                            bool setOnExtraOnly);
  INSTANTIATE_BC_TYPES(SpatialOps::structured::SVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::structured::XVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::structured::YVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::structured::ZVolField);
} // class BCHelper
