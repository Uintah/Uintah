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

#include "WasatchBCHelper.h"

//-- C++ Includes --//
#include <vector>
#include <iostream>

#include <boost/foreach.hpp>

//-- Uintah Includes --//
#include <Core/Grid/Patch.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/CellIterator.h> // Uintah::Iterator
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
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditionBase.h>


/**
 * \file    WasatchBCHelper.cc
 * \author  Tony Saad
 */

namespace WasatchCore {
  
  NSCBC::BCType get_nscbc_type(const BndTypeEnum& wasatchBndType)
  {
    switch (wasatchBndType) {
      case WALL:
        return NSCBC::WALL;
        break;
      case VELOCITY:
        return NSCBC::HARD_INFLOW;
        break;
      case OPEN:
      case OUTFLOW:
        return NSCBC::NONREFLECTING;
        break;
      default:
      {
        std::ostringstream msg;
        msg << "ERROR: An unsupported boundary type has been specified for the NSCBC boundary treatment. \n";
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
        break;
    }
  }
  
  SpatialOps::BCSide get_bc_side(const Uintah::Patch::FaceType face)
  {
    switch (face) {
      case Uintah::Patch::xminus:
      case Uintah::Patch::yminus:
      case Uintah::Patch::zminus:
        return SpatialOps::MINUS_SIDE;
        break;
      case Uintah::Patch::xplus:
      case Uintah::Patch::yplus:
      case Uintah::Patch::zplus:
        return SpatialOps::PLUS_SIDE;
        break;
      default:
      {
        std::ostringstream msg;
        msg << "ERROR: An invalid uintah face has been specified when tyring to apply bc \n";
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
        break;
    }
  }

  //============================================================================

  // This function returns true if the boundary condition is applied in the same direction
  // as the staggered field. For example, xminus/xplus on a XVOL field.
  NSCBC::Direction get_nscbc_dir( const Uintah::Patch::FaceType face ){
    switch (face) {
      case Uintah::Patch::xminus:
      case Uintah::Patch::xplus:
        return NSCBC::XDIR;
        break;
      case Uintah::Patch::yminus:
      case Uintah::Patch::yplus:
        return NSCBC::YDIR;
        break;
      case Uintah::Patch::zminus:
      case Uintah::Patch::zplus:
        return NSCBC::ZDIR;
        break;
      default:
      {
        std::ostringstream msg;
        msg << "ERROR: An invalid uintah face has been specified when tyring to apply bc \n";
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
        break;
    }
  }
  
  
  // This function returns true if the boundary condition is applied in the same direction
  // as the staggered field. For example, xminus/xplus on a XVOL field.
  template <typename MomDirT>
  NSCBC::Direction get_mom_dir(){}
  
  template<>
  NSCBC::Direction get_mom_dir<SpatialOps::XDIR>(){return NSCBC::XDIR;}
  
  template<>
  NSCBC::Direction get_mom_dir<SpatialOps::YDIR>(){return NSCBC::YDIR;}
  
  template<>
  NSCBC::Direction get_mom_dir<SpatialOps::ZDIR>(){return NSCBC::ZDIR;}

  //============================================================================

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

  //************************************************************************************************
  //
  //                          IMPLEMENTATION
  //
  //************************************************************************************************
  
  //------------------------------------------------------------------------------------------------

  WasatchBCHelper::WasatchBCHelper( const Uintah::LevelP& level,
                                   Uintah::SchedulerP& sched,
                                   const Uintah::MaterialSet* const materials,
                                   const PatchInfoMap& patchInfoMap,
                                   GraphCategories& grafCat,
                                   BCFunctorMap& bcFunctorMap,
                                   Uintah::ProblemSpecP wasatchSpec)
  : BCHelper(level, sched, materials),
    patchInfoMap_(patchInfoMap),
    bcFunctorMap_(bcFunctorMap),
    grafCat_     (grafCat)
  {
    Uintah::BBox b;
    level->getInteriorSpatialRange(b);
    const Uintah::Vector l = b.max() - b.min();
    
    nscbcSpec_.lx = l.x();
    nscbcSpec_.ly = l.y();
    nscbcSpec_.lz = l.z();
    nscbcSpec_.pFar = 101325.0;
    nscbcSpec_.enableNSCBC = false;
    
    // far field pressure
    if (wasatchSpec->findBlock("NSCBC")) {
      Uintah::ProblemSpecP nscbcXMLSpec = wasatchSpec->findBlock("NSCBC");
      double pFar = 101325.0;
      nscbcXMLSpec->getAttribute("pfarfield", pFar);
      nscbcSpec_.pFar = pFar;
      nscbcSpec_.enableNSCBC = true;
    }
  }

  //------------------------------------------------------------------------------------------------

  WasatchBCHelper::~WasatchBCHelper()
  {}
  
  //------------------------------------------------------------------------------------------------
  
  template<typename SrcT, typename TargetT>
  void WasatchBCHelper::create_dummy_dependency( const Expr::Tag& attachDepToThisTag,
                                          const Expr::TagList dependencies,
                                          const Category taskCat)
  {
    Expr::ExpressionFactory& factory = *(grafCat_[taskCat]->exprFactory);
    std::string phiName = attachDepToThisTag.name();
    
    // check if we already have an entry for phiname
    BCFunctorMap::iterator iter = bcFunctorMap_.find(phiName);
    size_t nMods = 0;

    Expr::Tag dummyTag( phiName + "_dummy_dependency", Expr::STATE_NONE );
    
    if( iter != bcFunctorMap_.end() ){
      (*iter).second.insert(dummyTag.name());
    }
    else {
      BCFunctorMap::mapped_type functorSet;
      nMods = (*iter).second.size();
      std::ostringstream msg;
      msg << nMods;
      dummyTag.reset_name( phiName + "_dummy_dependency_" + msg.str() );
      functorSet.insert( dummyTag.name() );
      bcFunctorMap_.insert( BCFunctorMap::value_type(phiName,functorSet) );
    }
    
    // if the dependency was already added then return
    if( factory.have_entry(dummyTag) ){
      return;
    }
    
    // register the null dependency
    typedef typename NullExpression<SrcT, TargetT>::Builder NullExpr;
    factory.register_expression( scinew NullExpr(dummyTag, dependencies), true );
  }
  
  //------------------------------------------------------------------------------------------------
  
  template <typename FieldT>
  void WasatchBCHelper::apply_boundary_condition( const Expr::Tag& varTag,
                                                  const Category& taskCat,
                                                  const bool setOnExtraOnly )

  {            
    using namespace std;
    const string& fieldName = varTag.name();
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
                    DBGBC << "dummy functor = " << modTag << std::endl;
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
              const Uintah::IntVector sciPatchCellOffset = patch->getExtraCellLowIndex(1);
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
              modExpr.set_bnd_type(myBndSpec.type);
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
              modExpr.set_spatial_mask      ( this->get_spatial_mask<FieldT>   (myBndSpec,patchID) );
              modExpr.set_svol_spatial_mask ( this->get_spatial_mask<SVolField>(myBndSpec,patchID) );
              modExpr.set_interior_svol_spatial_mask ( this->get_spatial_mask<SVolField>(myBndSpec,patchID, true) );
            }
          }
        }
      }
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void WasatchBCHelper::synchronize_pressure_expression()
  {
    Expr::ExpressionFactory& factory = *(grafCat_[ADVANCE_SOLUTION]->exprFactory);
    const Expr::Tag& pTag = TagNames::self().pressure;
    if (!factory.have_entry( pTag )) return;
    BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() ) {
      // loop over every patch in the patch subset
      BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() ) {
        const int patchID = patch->getID();
        // go through the patch ids and pass the WasatchBCHelper to the pressure expression
        Pressure& pexpr = dynamic_cast<Pressure&>( factory.retrieve_expression( pTag, patchID, true ) );
        pexpr.set_bchelper(this); // add the bchelper class to the pressure expression on ALL patches
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  
  void WasatchBCHelper::update_pressure_matrix( Uintah::CCVariable<Uintah::Stencil7>& pMatrix,
                                         const SVolField* const volFrac,
                                         const Uintah::Patch* patch )
  {
    const int patchID = patch->getID();
    const Expr::Tag& pTag = TagNames::self().pressure;

    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second; // get the boundary specification
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pTag.name()); // get the bc spec - we will check if the user specified anything for pressure here
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
          Uintah::Stencil7& coefs = pMatrix[*bndMask - unitNormal];
          
          // if we are inside a solid, then don't do anything because we already handle this in the pressure expression
          if( volFrac ){
            const Uintah::IntVector iCell = *bndMask - unitNormal - patch->getExtraCellLowIndex(1);
            const SpatialOps::IntVec iiCell(iCell.x(), iCell.y(), iCell.z() );
            if ((*volFrac)(iiCell) < 1.0)
              continue;
          }
          
          //
          switch(myBndSpec.face){
            case Uintah::Patch::xminus: coefs.p += sign*std::abs(coefs.w); coefs.w = 0.0; break;
            case Uintah::Patch::xplus : coefs.p += sign*std::abs(coefs.e); coefs.e = 0.0; break;
            case Uintah::Patch::yminus: coefs.p += sign*std::abs(coefs.s); coefs.s = 0.0; break;
            case Uintah::Patch::yplus : coefs.p += sign*std::abs(coefs.n); coefs.n = 0.0; break;
            case Uintah::Patch::zminus: coefs.p += sign*std::abs(coefs.b); coefs.b = 0.0; break;
            case Uintah::Patch::zplus : coefs.p += sign*std::abs(coefs.t); coefs.t = 0.0; break;
            default:                                                                      break;
          }
        }
      }
    }
  }
  
  //------------------------------------------------------------------------------------------------
  
  void WasatchBCHelper::apply_pressure_bc( SVolField& pressureField,
                                    const Uintah::Patch* patch )
  {
    typedef std::vector<SpatialOps::IntVec> MaskT;
    
    const int patchID = patch->getID();
    const Expr::Tag& pTag = TagNames::self().pressure;

    const Uintah::Vector res = patch->dCell();
    const double dx = res[0];
    const double dy = res[1];
    const double dz = res[2];

    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pTag.name());
      
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
  
  void WasatchBCHelper::update_pressure_rhs( SVolField& pressureRHS,
                                      const Uintah::Patch* patch )
  {
    typedef std::vector<SpatialOps::IntVec> MaskT;
    
    const int patchID = patch->getID();
    const Expr::Tag& pTag = TagNames::self().pressure;

    const Uintah::Vector res = patch->dCell();
    const double dx = res[0];
    const double dy = res[1];
    const double dz = res[2];
    
    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;
      const BndCondSpec* myBndCondSpec = bndSpecPair.second.find(pTag.name());
      
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
  
  template <typename MomDirT>
  void WasatchBCHelper::setup_nscbc(const BndSpec& myBndSpec, NSCBC::TagManager nscbcTagMgr, const int jobid)
  {
    if (!do_nscbc()) return;
    using namespace std;
    string bndName = myBndSpec.name;
    Expr::ExpressionFactory& factory = *(grafCat_[ADVANCE_SOLUTION]->exprFactory);
    typedef SVolField FieldT;
    
    bool do2, do3;
    const Expr::Tag& u = nscbcTagMgr[NSCBC::U];
    const Expr::Tag& v = nscbcTagMgr[NSCBC::V];
    const Expr::Tag& w = nscbcTagMgr[NSCBC::W];
    const bool dox = ( u != Expr::Tag() );
    const bool doy = ( v != Expr::Tag() );
    const bool doz = ( w != Expr::Tag() );
    double length = 1.0;
    
    NSCBC::Direction dir = get_nscbc_dir(myBndSpec.face);
    switch ( dir ) {
      case NSCBC::XDIR:
        do2 = doy;
        do3 = doz;
        length = nscbcSpec_.lx;
        break;
      case NSCBC::YDIR:
        do2 = dox;
        do3 = doz;
        length = nscbcSpec_.ly;
        break;
      case NSCBC::ZDIR:
        do2 = dox;
        do3 = doy;
        length = nscbcSpec_.lz;
        break;
      default:
        break;
    }
    
    const double gasConstant = 8314.459848;  // universal R = J/(kmol K).
    std::vector<double> mw = {28.966}; // we will need a vector of molecular weights---------
    BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() )
    {
      BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() )
      {
        const int patchID = patch->getID();
        const string strPatchID = number_to_string(patchID) + number_to_string(jobid);
        //_____________________________________________________________________________________
        // check if we have this patchID in the list of patchIDs
        if( myBndSpec.has_patch(patchID) ){
          const SpatialOps::SpatialMask<FieldT>& mask = *this->get_spatial_mask<FieldT>(myBndSpec,patchID,true);
          NSCBC::NSCBCInfo<FieldT> nscbcInfo( mask,
                                              get_bc_side(myBndSpec.face),
                                              get_nscbc_dir(myBndSpec.face),
                                              get_nscbc_type(myBndSpec.type),
                                              strPatchID,
                                              nscbcSpec_.pFar,
                                              length );
          
          NSCBC::BCBuilder<FieldT>* nscbcBuilder = new NSCBC::BCBuilder<FieldT>(nscbcInfo, mw, gasConstant, nscbcTagMgr, do2, do3, patchID);
          if ( nscbcBuildersMap_.find(bndName) != nscbcBuildersMap_.end() ) {
            if (nscbcBuildersMap_.find(bndName)->second.find(patchID) == nscbcBuildersMap_.find(bndName)->second.end()) {
              nscbcBuildersMap_.find(bndName)->second.insert( pair< int, NSCBC::BCBuilder<FieldT>* > (patchID, nscbcBuilder)       );
            } else {
              continue;
            }
            
          } else {
            PatchIDNSCBCBuilderMapT patchIDBuilderMap;
            patchIDBuilderMap.insert( pair< int, NSCBC::BCBuilder<FieldT>* > (patchID, nscbcBuilder)       );
            nscbcBuildersMap_.insert( pair< string, PatchIDNSCBCBuilderMapT >(bndName, patchIDBuilderMap ) );
          }
          //
        }
      }
    }
  }
  //------------------------------------------------------------------------------------------------
  
  void WasatchBCHelper::apply_nscbc_boundary_condition(const Expr::Tag& varTag,
                                                       const NSCBC::TransportVal& quantity,
                                                       const Category& taskCat)
  {
    if (!do_nscbc()) return;
    using namespace std;
    typedef SVolField FieldT;
    const string& fieldName = varTag.name();
    Expr::ExpressionFactory& factory = *(grafCat_[taskCat]->exprFactory);
    
    BOOST_FOREACH( const BndMapT::value_type bndSpecPair, bndNameBndSpecMap_ )
    {
      const BndSpec& myBndSpec = bndSpecPair.second;      
      {
        BOOST_FOREACH( const Uintah::PatchSubset* const patches, localPatches_->getVector() )
        {
          BOOST_FOREACH( const Uintah::Patch* const patch, patches->getVector() )
          {
            const int patchID = patch->getID();
            const string strPatchID = number_to_string(patchID);
            //_____________________________________________________________________________________
            // check if we have this patchID in the list of patchIDs
            if( myBndSpec.has_patch(patchID) ) {
              string bndName = myBndSpec.name;
              NSCBC::BCBuilder<FieldT>* nscbc = nscbcBuildersMap_.find(bndName)->second.find(patchID)->second;
              nscbc->attach_rhs_modifier( factory, varTag, quantity, -1 );
            }
          }
        }
      }
    }

//    {
//      const Expr::Tag t1("RP_XFace_Plus_NonreflectingFlow30", Expr::STATE_NONE);
//      const Expr::Tag t2("RP_XFace_Minus_NonreflectingFlow00", Expr::STATE_NONE);
//      
//      
//      typedef typename Expr::ConstantExpr<FieldT>::Builder ConstExprT;
//      
//      if( !( factory.have_entry(t1) ) ){
//        
//        factory.register_expression(new ConstExprT( t1, 0.0 ), false);
//        create_dummy_dependency<SVolField, SVolField>(Expr::Tag("x-mom_rhs",Expr::STATE_NONE), tag_list(t1), ADVANCE_SOLUTION);
//      }
//      
//      if( !( factory.have_entry(t2) ) ){
//        
//        factory.register_expression(new ConstExprT( t2, 0.0 ), false);
//        create_dummy_dependency<SVolField, SVolField>(Expr::Tag("x-mom_rhs",Expr::STATE_NONE), tag_list(t2), ADVANCE_SOLUTION);
//      }
//      
//    }
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
            DBGBC << "nscbc patchID = " << patchID << std::endl;
            BCFunctorMap::iterator iter = bcFunctorMap_.begin();
            while ( iter != bcFunctorMap_.end() ) {
              string functorPhiName = (*iter).first;
              DBGBC << "nscbc functor PhiName = " << functorPhiName << " fieldname = " << fieldName << std::endl;
              if ( functorPhiName.compare(fieldName) == 0 ) {
                // get the functor set associated with this field
                BCFunctorMap::mapped_type::const_iterator functorIter = (*iter).second.begin();
                while( functorIter != (*iter).second.end() ){
                  const string& functorName = *functorIter;
                  const Expr::Tag modTag = Expr::Tag(functorName,Expr::STATE_NONE);
                  DBGBC << "nscbc functor = " << modTag << std::endl;

                  if (factory.have_entry(modTag)) {
                    DBGBC << "dummy nscbc functor = " << modTag << std::endl;
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

  }

  //------------------------------------------------------------------------------------------------

  //==========================================================================
  // Explicit template instantiation for supported versions of this class
  #include <spatialops/structured/FVStaggered.h>

#define INSTANTIATE_BC_TYPES(VOLT) \
  template void WasatchBCHelper::apply_boundary_condition< VOLT >( const Expr::Tag& varTag,            \
                                                            const Category& taskCat,            \
                                                            const bool setOnExtraOnly);         \
  template void WasatchBCHelper::create_dummy_dependency< VOLT, SpatialOps::SVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void WasatchBCHelper::create_dummy_dependency< VOLT, SpatialOps::XVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void WasatchBCHelper::create_dummy_dependency< VOLT, SpatialOps::YVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void WasatchBCHelper::create_dummy_dependency< VOLT, SpatialOps::ZVolField >( const Expr::Tag& attachDepToThisTag, \
                                                                                  const Expr::TagList dependencies,    \
                                                                                  const Category taskCat );            \
  template void WasatchBCHelper::create_dummy_dependency< SpatialOps::SingleValueField, VOLT >( const Expr::Tag& attachDepToThisTag, \
                                                                                 const Expr::TagList dependencies,     \
                                                                                 const Category taskCat );

  INSTANTIATE_BC_TYPES(SpatialOps::SVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::SSurfXField);
  INSTANTIATE_BC_TYPES(SpatialOps::SSurfYField);
  INSTANTIATE_BC_TYPES(SpatialOps::SSurfZField);
  INSTANTIATE_BC_TYPES(SpatialOps::XVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::YVolField);
  INSTANTIATE_BC_TYPES(SpatialOps::ZVolField);
  
  template void WasatchBCHelper::apply_boundary_condition< ParticleField >( const Expr::Tag& varTag,
                                                                            const Category& taskCat,
                                                                            const bool setOnExtraOnly);
  
  template void WasatchBCHelper::setup_nscbc<SpatialOps::XDIR>(const BndSpec& myBndSpec, NSCBC::TagManager nscbcTagMgr, const int jobid);
  template void WasatchBCHelper::setup_nscbc<SpatialOps::YDIR>(const BndSpec& myBndSpec, NSCBC::TagManager nscbcTagMgr, const int jobid);
  template void WasatchBCHelper::setup_nscbc<SpatialOps::ZDIR>(const BndSpec& myBndSpec, NSCBC::TagManager nscbcTagMgr, const int jobid);


} // class WasatchBCHelper
