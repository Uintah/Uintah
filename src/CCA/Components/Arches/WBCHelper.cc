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

#include <CCA/Components/Arches/WBCHelper.h>

//-- C++ Includes --//
#include <vector>
#include <iostream>

#include <boost/foreach.hpp>

//-- Uintah Includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/CellIterator.h> // Uintah::Iterator
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <CCA/Ports/Scheduler.h>

//-- Debug Stream --//
#include <Core/Util/DebugStream.h>

static Uintah::DebugStream dbgbc("ARCHES_BC", false);
#define DBC_BC_ON  dbgbc.active()
#define DBGBC  if( DBC_BC_ON  ) dbgbc

using namespace Uintah;

/**
 * \file    WBCHelper.cc
 * \author  Tony Saad
 */

// Given a string BC type (Dirichlet, Neumann,...), this function returns a BndCondTypeEnum
// of supported boundary condition types
BndCondTypeEnum select_bc_type_enum( const std::string& bcTypeStr )
{
  if      ( bcTypeStr == "Dirichlet" )     return DIRICHLET;
  else if ( bcTypeStr == "Neumann" )       return NEUMANN;
  else                                     return CUSTOM;
}

// Given a string boundary type (Wall, Velocity, Outflow,...), this function returns a BndTypeEnum
// of supported boundary types
BndTypeEnum select_bnd_type_enum( const std::string& bndTypeStr )
{
  if      ( bndTypeStr == "Wall"     )  return WALL;
  else if ( bndTypeStr == "Inlet" )     return INLET;
  else if ( bndTypeStr == "Outflow" )   return OUTLET;
  else if ( bndTypeStr == "Pressure" )  return PRESSURE;
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

// Given a BndTypeEnum (WALL, INLET,...), this function returns a string
// of supported boundary types
const std::string bnd_type_enum_to_string( const BndTypeEnum bndTypeEnum )
{
  switch (bndTypeEnum) {
    case WALL:
      return "Wall";
      break;
    case INLET:
      return "Inlet";
      break;
    case OUTLET:
      return "Outflow";
      break;
    case PRESSURE:
      return "Pressure";
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

//****************************************************************************
/**
 *
 *  \brief Helps with staggered fields.
 *
 */
//****************************************************************************

void pack_uintah_iterator( const Uintah::Patch::FaceType& face,
                           const Uintah::Patch* const patch,
                           Uintah::Iterator& bndIter,
                           BoundaryIterators& myBndIters )
{
  //bool plusEdge[3];
  //bool minusEdge[3];

  //minusEdge[0] = patch->getBCType(Uintah::Patch::xminus) != Uintah::Patch::Neighbor;
  //plusEdge [0] = patch->getBCType(Uintah::Patch::xplus ) != Uintah::Patch::Neighbor;
  //minusEdge[1] = patch->getBCType(Uintah::Patch::yminus) != Uintah::Patch::Neighbor;
  //plusEdge [1] = patch->getBCType(Uintah::Patch::yplus ) != Uintah::Patch::Neighbor;
  //minusEdge[2] = patch->getBCType(Uintah::Patch::zminus) != Uintah::Patch::Neighbor;
  //plusEdge [2] = patch->getBCType(Uintah::Patch::zplus ) != Uintah::Patch::Neighbor;

  // int i=-1, j=-1;
  // switch (face) {
  //   case Uintah::Patch::xminus:
  //   case Uintah::Patch::xplus: i=1; j=2; break;
  //   case Uintah::Patch::yminus:
  //   case Uintah::Patch::yplus: i=0; j=2; break;
  //   case Uintah::Patch::zminus:
  //   case Uintah::Patch::zplus: i=0; j=1; break;
  //   default:{
  //     std::ostringstream msg;
  //     msg << "ERROR: invalid face specification encountered\n"
  //         << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
  //     throw std::runtime_error( msg.str() );
  //   }
  // }

  // save pointer to the Uintah iterator. This will be needed for expressions that require access to the
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
  const Uintah::IntVector idxHi = patch->getCellHighIndex() - Uintah::IntVector(1,1,1);// - patchCellOffset;
  const Uintah::IntVector idxLo = patch->getCellLowIndex();

}

//// This function returns true if the boundary condition is applied in the same direction
//// as the staggered field. For example, xminus/xplus on a staggered X field.
//template <typename FieldT>
//bool is_staggered_normal( const Uintah::Patch::FaceType face ){
//  const Direction staggeredLocation = get_staggered_location<FieldT>();
//  switch (staggeredLocation) {
//    case XDIR:
//      return ( (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus)); break;
//    case YDIR:
//      return ( (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus)); break;
//    case ZDIR:
//      return ( (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus)); break;
//    default: return false; break;
//  }
//  return false;
//}

//// This function returns true if the boundary condition is applied in the same direction
//// as the staggered field. For example, xminus/xplus on a XVOL field.
//template<typename FieldT>
//bool is_plus_side( const Uintah::Patch::FaceType face ){
//  const Direction staggeredLocation = get_staggered_location<FieldT>();
//  switch (staggeredLocation) {
//    case XDIR: return (face==Uintah::Patch::xplus);  break;
//    case YDIR: return (face==Uintah::Patch::yplus);  break;
//    case ZDIR: return (face==Uintah::Patch::zplus);  break;
//    default:   return false; break;
//  }
//  return false;
//}

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
    return nullptr;
  }
}

// find the BCSpec associated with a given variable name - non-const version
BndCondSpec* BndSpec::find_to_edit(const std::string& varName)
{
  std::vector<BndCondSpec>::iterator it = std::find(bcSpecVec.begin(), bcSpecVec.end(), varName);
  if (it != bcSpecVec.end()) {
    return &(*it);
  } else {
    return nullptr;
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
  cout << "Boundary: " << name << " face: " << face << " BndType: " << type << " area: " << area << endl;
  for (vector<BndCondSpec>::const_iterator it=bcSpecVec.begin(); it != bcSpecVec.end(); ++it) {
    (*it).print();
  }
}

//============================================================================


//************************************************************************************************
//
//                          IMPLEMENTATION
//
//************************************************************************************************

//------------------------------------------------------------------------------------------------

WBCHelper::WBCHelper( const Uintah::LevelP& level,
                      Uintah::SchedulerP& sched,
                      const Uintah::MaterialSet* const materials,
                      ProblemSpecP arches_spec )
: materials_   (materials   ), m_arches_spec(arches_spec)
{
  const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
  const Uintah::PatchSubset* const localPatches = allPatches->getSubset( Uintah::Parallel::getMPIRank() );
  localPatches_ = new Uintah::PatchSet;
  localPatches_->addEach( localPatches->getVector() );
  const int ilvl = level->getID();
  parse_boundary_conditions(ilvl);
}

  //------------------------------------------------------------------------------------------------

WBCHelper::~WBCHelper()
{

  delete localPatches_;
  delete_area_labels();

}

  //------------------------------------------------------------------------------------------------

void WBCHelper::add_boundary_condition( const std::string& bndName,
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

void WBCHelper::add_boundary_condition( const BndCondSpec& bcSpec )
{
  using namespace std;
  for ( auto bndPair = bndNameBndSpecMap_.begin(); bndPair != bndNameBndSpecMap_.end(); bndPair++ ){
    add_boundary_condition((*bndPair).first, bcSpec);
  }
}

//------------------------------------------------------------------------------------------------

void WBCHelper::add_boundary( const std::string&     bndName,
                             Uintah::Patch::FaceType face,
                             const BndTypeEnum&      bndType,
                             const BndEdgeType&      bndEdgeType,
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
    BndSpec myBndSpec = {bndName, face, bndType, bndEdgeType, 0.0, std::vector<int>(1, patchID), myPBndSpec };
    bndNameBndSpecMap_.insert( BndMapT::value_type(bndName, myBndSpec) );
  }
}

//------------------------------------------------------------------------------------------------

void WBCHelper::add_auxiliary_boundary_condition( const std::string& srcVarName,
                                                 const std::string& newVarName,
                                                 const double newValue,
                                                 const BndCondTypeEnum newBCType )
{
  BndCondSpec newBCSpec = {newVarName, "none", newValue, newBCType, DOUBLE_TYPE};
  add_auxiliary_boundary_condition(srcVarName, newBCSpec);
}

//------------------------------------------------------------------------------------------------

void WBCHelper::add_auxiliary_boundary_condition(const std::string& srcVarName,
                                                BndCondSpec bcSpec)
{
  for ( auto bndSpecPair = bndNameBndSpecMap_.begin(); bndSpecPair != bndNameBndSpecMap_.end();
        bndSpecPair++ ){
    BndSpec& myBndSpec = (*bndSpecPair).second;
    const BndCondSpec* myBndCondSpec = myBndSpec.find(srcVarName);
    if (myBndCondSpec) {
      add_boundary_condition(myBndSpec.name, bcSpec);
    }
  }
}

//------------------------------------------------------------------------------------------------

Uintah::Iterator&
WBCHelper::get_uintah_extra_bnd_mask( const BndSpec& myBndSpec,
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
  msg << "ERROR: It looks like you were trying to grab a Uintah boundary iterator that doesn't exist! "
  << "This could be caused by requesting an iterator for a boundary/patch combination that is inconsistent with your input. "
  << "Otherwise, this is likely a major bug that needs to be addressed by a core Arches developer." << std::endl;
  throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );

}

//------------------------------------------------------------------------------------------------

void WBCHelper::print() const
{
  for ( auto bndNameBCSpecPair = bndNameBndSpecMap_.begin();
        bndNameBCSpecPair != bndNameBndSpecMap_.end(); bndNameBCSpecPair++ ){
    (*bndNameBCSpecPair).second.print();
  }
}

//------------------------------------------------------------------------------------------------

void WBCHelper::add_boundary_mask( const BoundaryIterators& myIters,
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

void WBCHelper::parse_boundary_conditions(const int ilvl)
{
  // create area labels for all areas
  ProblemSpecP db_root = m_arches_spec->getRootNode();
  ProblemSpecP db_bc   = db_root->findBlock("Grid")->findBlock("BoundaryConditions");
  ProblemSpecP db_bc_arches = m_arches_spec->findBlock("BoundaryConditions");
  bool force_area_calc = false;
  if ( db_bc_arches ){
    if ( db_bc_arches->findBlock("force_area_calc") ){
      force_area_calc = true;
    }
  }

  //This will contain all the bc's everywhere. We will cherry pick from it below to only
  // store information here per patch.

  if ( db_bc ) {
    for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ) {
      bool found_mass_flux = false;
      for ( ProblemSpecP db_bc = db_face->findBlock("BCType"); db_bc != nullptr; db_bc = db_bc->findNextBlock("BCType") ) {
        std::string var;
        db_bc->getAttribute("var", var );
        if ( var == "MassFlowInlet" || var == "Swirl" ){
          found_mass_flux = true;
        }
        if ( force_area_calc ){
          found_mass_flux = true;
        }
      }
      std::string faceName;
      db_face->getAttribute("name",faceName);
      std::string level_index = std::to_string(ilvl);
      const std::string faceAreaName = faceName+"_"+level_index+"_area";
      if ( found_mass_flux ){
        create_new_area_label( faceAreaName );
      }
    }
  }

  using namespace std;
  // loop over the material set
  for ( auto i_matSubSet = (materials_->getVector()).begin();
        i_matSubSet != (materials_->getVector()).end(); i_matSubSet++ ){

    const Uintah::MaterialSubset* matSubSet = *i_matSubSet;

    // loop over materials
    for( int im=0; im < matSubSet->size(); ++im ) {

      const int materialID = matSubSet->get(im);

      // loop over local patches
      for ( auto i_patches = (localPatches_->getVector()).begin();
            i_patches != (localPatches_->getVector()).end(); i_patches++ ){

        const Uintah::PatchSubset* patches = *i_patches;

        // loop over every patch in the patch subset
        for ( auto i_patch = (patches->getVector()).begin();
              i_patch != (patches->getVector()).end(); i_patch++ ){

          const Patch* patch = *i_patch;

          const int patchID = patch->getID();
          DBGBC << "Patch ID = " << patchID << std::endl;

          std::vector<Uintah::Patch::FaceType> bndFaces;
          patch->getBoundaryFaces(bndFaces);

          for ( auto i_face = bndFaces.begin(); i_face != bndFaces.end(); i_face++ ){

            const Uintah::Patch::FaceType face = *i_face;

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
              //const std::string bndAreaName = bndName+"_area";
              //create_new_area_label( bndAreaName );

              if (bndName.compare("NotSet")==0) {
                std::ostringstream msg;
                msg << "ERROR: It looks like you have not set a name for one of your boundary conditions! "
                    << "You MUST specify a name for your <Face> spec boundary condition. Please revise your input file." << std::endl;
                throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
              }
              DBGBC << " boundary name = " << bndName << std::endl;
              DBGBC << " geom bndtype  = " << thisGeom->getBndType() << std::endl;
              BndTypeEnum bndType = select_bnd_type_enum(thisGeom->getBndType());
              add_boundary( bndName, face, bndType, EDGE, patchID, thisGeom->getParticleBndSpec() );
              DBGBC << " boundary type = " << bndType << std::endl;

              //__________________________________________________________________________________
              Uintah::Iterator bndIter; // allocate iterator
              // get the iterator for the extracells for this child
              thisGeom->getCellFaceIterator(bndIter);

              BoundaryIterators myIters;
              DBGBC << " Size of uintah iterator for boundary: " << bndName << " = " << bndIter.size() << std::endl;
              pack_uintah_iterator(face, patch, bndIter, myIters); // store uintah iterator
              // store a pointer to the list of particle index that are near this boundary.
              //myIters.particleIdx = Uintah::ParticlesHelper::get_boundary_particles(bndName,patchID);
              add_boundary_mask( myIters, bndName, patchID );

              //__________________________________________________________________________________
              // Now, each BCGeomObject has BCData associated with it. This BCData contains the list
              // of variables and types (Dirichlet, etc...), and values that the user specified
              // through the input file!
              Uintah::BCData bcData;
              thisGeom->getBCData(bcData);

              for ( auto i_bndCondBase = (bcData.getBCData()).begin(); i_bndCondBase !=
                    (bcData.getBCData()).end(); i_bndCondBase++ ){

                Uintah::BoundCondBase* bndCondBase = *i_bndCondBase;

                const std::string varName     = bndCondBase->getBCVariable();
                const BndCondTypeEnum atomBCTypeEnum = select_bc_type_enum(bndCondBase->getBCType());

                DBGBC << " bc variable = " << varName << std::endl
                << " bc type = "     << atomBCTypeEnum << std::endl;

                double doubleVal=0.0;
                std::string functorName="none";
                BCValueTypeEnum bcValType=INVALID_TYPE;

                if ( atomBCTypeEnum == DIRICHLET || atomBCTypeEnum == NEUMANN ){

                  if ( bndCondBase->getValueType() == Uintah::BoundCondBase::VECTOR_TYPE ){
                    // This is a hack to accomidate the Velocity BC in old arches.
                  } else {
                    // doubles
                    const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                    doubleVal = new_bc->getValue();
                    bcValType = DOUBLE_TYPE;
                  }
                } else {
                  // functors
                  if ( bndCondBase->getValueType() == Uintah::BoundCondBase::DOUBLE_TYPE ){
                    const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                    functorName = new_bc->getType();
                    bcValType = FUNCTOR_TYPE;

                    DBGBC << " functor name = " << functorName << std::endl;
                  } else if ( bndCondBase->getValueType() == Uintah::BoundCondBase::STRING_TYPE ){
                    const Uintah::BoundCond<std::string>* const new_bc = dynamic_cast<const Uintah::BoundCond<std::string>*>(bndCondBase);
                    functorName = new_bc->getType();
                    bcValType = FUNCTOR_TYPE;

                    DBGBC << " functor name = " << functorName << std::endl;
                  } else if ( bndCondBase->getValueType() == Uintah::BoundCondBase::VECTOR_TYPE ){

                    // do nothing currently... need to evaluate?
                    // This is added to keep the old Arches from erroring out...

                  } else {
                    std::ostringstream msg;
                    msg << "ERROR: It looks like you have specified an unsupported datatype value for boundary " << bndName << ". "
                        << "Supported datatypes are: double and string." << std::endl;
                    throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                  }
                }

                const BndCondSpec bndCondSpec = { varName, functorName, doubleVal, atomBCTypeEnum,
                                                  bcValType };

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
                add_boundary( bndName, face_side, bndType, INTERIOR, patchID, thisGeom->getParticleBndSpec() );
                DBGBC << " boundary type = " << bndType << std::endl;

                //__________________________________________________________________________________
                Uintah::Iterator bndIter; // allocate iterator
                // get the iterator for the extracells for this child
                thisGeom->getCellFaceIterator(bndIter);

                BoundaryIterators myIters;
                DBGBC << " Size of uintah iterator for boundary: " << bndName << " = " << bndIter.size() << std::endl;
                pack_uintah_iterator(face_side, patch, bndIter, myIters); // store uintah iterator
                // store a pointer to the list of particle index that are near this boundary.
                //myIters.particleIdx = Uintah::ParticlesHelper::get_boundary_particles(bndName,patchID);
                add_boundary_mask( myIters, bndName, patchID );

                //__________________________________________________________________________________
                // Now, each BCGeomObject has BCData associated with it. This BCData contains the list
                // of variables and types (Dirichlet, etc...), and values that the user specified
                // through the input file!
                Uintah::BCData bcData;
                thisGeom->getBCData(bcData);

                for ( auto i_bndCondBase = (bcData.getBCData()).begin(); i_bndCondBase !=
                      (bcData.getBCData()).end(); i_bndCondBase++ ){

                  Uintah::BoundCondBase* bndCondBase = *i_bndCondBase;

                  const std::string varName     = bndCondBase->getBCVariable();
                  const BndCondTypeEnum atomBCTypeEnum = select_bc_type_enum(bndCondBase->getBCType());

                  DBGBC << " bc variable = " << varName << std::endl
                        << " bc type = "     << atomBCTypeEnum << std::endl;

                  double doubleVal=0.0;
                  std::string functorName="none";
                  BCValueTypeEnum bcValType=INVALID_TYPE;

                  if ( atomBCTypeEnum == DIRICHLET || atomBCTypeEnum == NEUMANN ){

                    if ( bndCondBase->getValueType() == Uintah::BoundCondBase::VECTOR_TYPE ){
                      // This is a hack to accomidate the Velocity BC in old arches.
                    } else {
                      // doubles
                      const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                      doubleVal = new_bc->getValue();
                      bcValType = DOUBLE_TYPE;
                    }
                  } else {
                    // functors
                    if ( bndCondBase->getValueType() == Uintah::BoundCondBase::DOUBLE_TYPE ){
                      const Uintah::BoundCond<double>* const new_bc = dynamic_cast<const Uintah::BoundCond<double>*>(bndCondBase);
                      functorName = new_bc->getType();
                      bcValType = FUNCTOR_TYPE;

                      DBGBC << " functor name = " << functorName << std::endl;
                    } else if ( bndCondBase->getValueType() == Uintah::BoundCondBase::STRING_TYPE ){
                      const Uintah::BoundCond<std::string>* const new_bc = dynamic_cast<const Uintah::BoundCond<std::string>*>(bndCondBase);
                      functorName = new_bc->getType();
                      bcValType = FUNCTOR_TYPE;

                      DBGBC << " functor name = " << functorName << std::endl;
                    } else if ( bndCondBase->getValueType() == Uintah::BoundCondBase::VECTOR_TYPE ){

                      // do nothing currently... need to evaluate?
                      throw Uintah::ProblemSetupException("Error: Vector Type for BCs not recognized.", __FILE__, __LINE__);

                    } else {
                      std::ostringstream msg;
                      msg << "ERROR: It looks like you have specified an unsupported datatype value for boundary " << bndName << ". "
                          << "Supported datatypes are: double and string." << std::endl;
                      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
                    }
                  }

                  const BndCondSpec bndCondSpec = { varName, functorName, doubleVal,
                                                    atomBCTypeEnum, bcValType };

                  add_boundary_condition(bndName, bndCondSpec);

                }
              } // boundary child loop (note, a boundary child is what Wasatch thinks of as a boundary condition
            }
          }

          //put the functors associated with this patch into perma-storage
          //patch_to_functor_storage.insert(std::make_pair(patchID, bc_functor_storage));

        } // patch loop
      } // patch subset loop
    } // material loop
  } // material subset loop
}

//------------------------------------------------------------------------------------------------

const BndMapT& WBCHelper::get_boundary_information() const
{
  return bndNameBndSpecMap_;
}

//------------------------------------------------------------------------------------------------

BndMapT& WBCHelper::get_for_edit_boundary_information()
{
  return bndNameBndSpecMap_;
}

//------------------------------------------------------------------------------------------------

bool WBCHelper::has_boundaries() const
{
  return !bndNameBndSpecMap_.empty();
}

//------------------------------------------------------------------------------------------------

void WBCHelper::create_new_area_label( const std::string name ){
  auto iter = m_area_labels.find(name);
  if ( iter == m_area_labels.end() ){
    const VarLabel* label = VarLabel::create( name, ReductionVariable<double,
                                              Reductions::Sum<double> >::getTypeDescription());
    m_area_labels.insert(std::make_pair(name, label));
  }
}

void WBCHelper::delete_area_labels(){
  for (auto iter = m_area_labels.begin(); iter != m_area_labels.end(); iter++){
    VarLabel::destroy(iter->second);
  }
}

void WBCHelper::sched_computeBCAreaHelper( SchedulerP& sched,
                                const LevelP& level,
                                const MaterialSet* matls ){
    IntVector lo;
    IntVector hi;
    level->findInteriorCellIndexRange(lo,hi);

    Task* tsk = scinew Task( "WBCHelper::computeBCAreaHelper", this,
                             &WBCHelper::computeBCAreaHelper, lo, hi );

    for ( auto iter = m_area_labels.begin(); iter != m_area_labels.end(); iter++ ){
      tsk->computes( iter->second );
    }

    sched->addTask( tsk, level->eachPatch(), matls );
}


void WBCHelper::computeBCAreaHelper( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const IntVector lo,
                          const IntVector hi ){
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();
    const int ilvl = level->getID();

    Vector DX = patch->dCell();

    const BndMapT& my_map = get_boundary_information();
    for ( auto bndmap_iter = my_map.begin(); bndmap_iter != my_map.end();
          bndmap_iter++ ){

      BndSpec bc_spec = bndmap_iter->second;
      const std::string bc_name = bc_spec.name;
      const int pid = patch->getID();

      //this current bc is only valid for a subset of patches (potentially).
      // however, the current mpi rank may hold multiple patches. therefore,
      // only do the following if the bc_spec is valid on the current patch.
      for ( auto i_valid_pid = bc_spec.patchIDs.begin(); i_valid_pid != bc_spec.patchIDs.end();
            i_valid_pid++ ){

        if ( *i_valid_pid == pid ){

          double darea = 0;
          int i=-1; int j=-1;
          if ( bc_spec.face == Patch::xminus || bc_spec.face == Patch::xplus ){
            darea = DX.y() * DX.z();
            i = 1; j = 2;
          } else if ( bc_spec.face == Patch::yminus || bc_spec.face == Patch::yplus ){
            darea = DX.x() * DX.z();
            i = 2; j = 0;
          } else if ( bc_spec.face == Patch::zminus || bc_spec.face == Patch::zplus ){
            darea = DX.x() * DX.y();
            i = 0; j = 1;
          }

          //get the Uintah Iterator:
          Uintah::Iterator& grid_iter = get_uintah_extra_bnd_mask( bndmap_iter->second,
                                                                   patch->getID() );
          double area = 0.;
          for ( grid_iter.reset(); !grid_iter.done(); grid_iter++ ){

            //exclude edge cells
            if ( (*grid_iter)[i] != -1 && (*grid_iter)[i] != hi[i] ){
              if ( (*grid_iter)[j] != -1 && (*grid_iter)[j] != hi[j] ){
                area += darea;
              }
            }
          }

          std::string level_index = std::to_string(ilvl);
          auto bc_area_iter = m_area_labels.find(bc_name+"_"+level_index+"_area");
          if ( bc_area_iter != m_area_labels.end() ){
            new_dw->put(sum_vartype( area ), bc_area_iter->second );
          }

        }
      }
    }
  }
}

void WBCHelper::sched_bindBCAreaHelper( SchedulerP& sched,
                                        const LevelP& level,
                                        const MaterialSet* matls ){
    Task* tsk = scinew Task( "WBCHelper::bindBCAreaHelper", this,
                             &WBCHelper::bindBCAreaHelper );

    for ( auto iter = m_area_labels.begin(); iter != m_area_labels.end(); iter++ ){
      tsk->requires( Task::NewDW, iter->second );
    }

    sched->addTask( tsk, level->eachPatch(), matls );

}

void WBCHelper::bindBCAreaHelper( const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw ){
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    const Level* level = patch->getLevel();
    const int ilvl = level->getID();

    BndMapT& my_map = get_for_edit_boundary_information();
    for ( auto bndmap_iter = my_map.begin(); bndmap_iter != my_map.end();
          bndmap_iter++ ){

      BndSpec bc_spec = bndmap_iter->second;
      const std::string bc_name = bc_spec.name;

      std::string level_index = std::to_string(ilvl);
      auto bc_area_iter = m_area_labels.find(bc_name+"_"+level_index+"_area");

      if ( bc_area_iter != m_area_labels.end() ){
        sum_vartype area_temp;
        new_dw->get(area_temp, bc_area_iter->second );
        double area = area_temp;
        bndmap_iter->second.area = area;
      } else {
        bndmap_iter->second.area = 0.0;
      }

    }
  }
}
