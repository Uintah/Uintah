/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include "DualTimeMatrixManager.h"

#include <Core/Util/DebugStream.h>

#include <expression/dualtime/BlockImplicitBDFDualTimeIntegrator.h>
#include <expression/matrix-assembly/MatrixExpression.h> /* for debugging purposes */
#include <expression/matrix-assembly/SparseMatrix.h>


static Uintah::DebugStream dbgd("WASATCH_DUALTIME", false); // dual time diagnostics
#define dbg_dualtime_on dbgd.active() && Uintah::Parallel::getMPIRank() == 0
#define dbg_dualtime if( dbg_dualtime_on ) dbgd


namespace WasatchCore{


  void DualTimeMatrixInfo::set_mass_fraction_tags( const Expr::TagList tags )
  {
    massFractions = tags;
  }

  void DualTimeMatrixInfo::set_species_density_tags( const Expr::TagList tags )
  {
    speciesDensities = tags;
  }

  void DualTimeMatrixInfo::set_enthalpies( const Expr::TagList tags )
  {
    enthalpies = tags;
  }

  void DualTimeMatrixInfo::set_energies( const Expr::TagList tags )
  {
    energies = tags;
  }

  void DualTimeMatrixInfo::set_diffusivities( const Expr::TagList tags )
  {
    diffusivities = tags;
  }

  void DualTimeMatrixInfo::set_production_rates( const Expr::TagList tags )
  {
    productionRates = tags;
  }

  void DualTimeMatrixInfo::set_molecular_weights( const std::vector<double> mw )
  {
    molecularWeights = mw;
  }

  void DualTimeMatrixInfo::add_scalar_equation( const Expr::Tag solution_variable_tag,
                                                const Expr::Tag rhs_tag )
  {
    scalarVariables.push_back( solution_variable_tag );
    scalarRightHandSides.push_back( rhs_tag );
  }

  void DualTimeMatrixManager::print_tags()
  {
    dbg_dualtime << "- Checking matrix info: viscosity: " << info_.viscosity << "\n";
    dbg_dualtime << "- Checking matrix info: density: " << info_.density << "\n";
    dbg_dualtime << "- Checking matrix info: cpHeatCapacity: " << info_.cpHeatCapacity << "\n";
    dbg_dualtime << "- Checking matrix info: cvHeatCapacity: " << info_.cvHeatCapacity << "\n";
    dbg_dualtime << "- Checking matrix info: conductivity: " << info_.conductivity << "\n";
    dbg_dualtime << "- Checking matrix info: pressure: " << info_.pressure << "\n";
    dbg_dualtime << "- Checking matrix info: temperature: " << info_.temperature << "\n";
    dbg_dualtime << "- Checking matrix info: totalEnergy: " << info_.totalEnergy << "\n";
    dbg_dualtime << "- Checking matrix info: totalEnthalpy: " << info_.totalEnthalpy << "\n";
    dbg_dualtime << "- Checking matrix info: xCoord: " << info_.xCoord << "\n";
    dbg_dualtime << "- Checking matrix info: yCoord: " << info_.yCoord << "\n";
    dbg_dualtime << "- Checking matrix info: zCoord: " << info_.zCoord << "\n";
    dbg_dualtime << "- Checking matrix info: xMomentum: " << info_.xMomentum << "\n";
    dbg_dualtime << "- Checking matrix info: yMomentum: " << info_.yMomentum << "\n";
    dbg_dualtime << "- Checking matrix info: zMomentum: " << info_.zMomentum << "\n";
    dbg_dualtime << "- Checking matrix info: xVelocity: " << info_.xVelocity << "\n";
    dbg_dualtime << "- Checking matrix info: yVelocity: " << info_.yVelocity << "\n";
    dbg_dualtime << "- Checking matrix info: zVelocity: " << info_.zVelocity << "\n";
    dbg_dualtime << "- Checking matrix info: soundSpeed: " << info_.soundSpeed << "\n";
    dbg_dualtime << "- Checking matrix info: mmw: " << info_.mmw << "\n";
    dbg_dualtime << "- Checking matrix info: timeStepSize: " << info_.timeStepSize << "\n";
    dbg_dualtime << "- Checking matrix info: universalGasConstant: " << info_.universalGasConstant << "\n";

    dbg_dualtime << "- Checking matrix info: massFractions size: " << info_.massFractions.size() << "\n";
    dbg_dualtime << "- Checking matrix info: speciesDensities size: " << info_.speciesDensities.size() << "\n";
    dbg_dualtime << "- Checking matrix info: enthalpies size: " << info_.enthalpies.size() << "\n";
    dbg_dualtime << "- Checking matrix info: energies size: " << info_.energies.size() << "\n";
    dbg_dualtime << "- Checking matrix info: diffusivities size: " << info_.diffusivities.size() << "\n";
    dbg_dualtime << "- Checking matrix info: productionRates size: " << info_.productionRates.size() << "\n";
    dbg_dualtime << "- Checking matrix info: scalarRightHandSides size: " << info_.scalarRightHandSides.size() << "\n";
    dbg_dualtime << "- Checking matrix info: scalarVariables size: " << info_.scalarVariables.size() << "\n";
    dbg_dualtime << "- Checking matrix info: molecularWeights size: " << info_.molecularWeights.size() << "\n";

    dbg_dualtime << "- Checking matrix info: doSpecies: " << info_.doSpecies << "\n";
    dbg_dualtime << "- Checking matrix info: doCompressible: " << info_.doCompressible << "\n";
    dbg_dualtime << "- Checking matrix info: doInviscidImplicit: " << info_.doImplicitInviscid << "\n";
    dbg_dualtime << "- Checking matrix info: doX: " << info_.doX << "\n";
    dbg_dualtime << "- Checking matrix info: doY: " << info_.doY << "\n";
    dbg_dualtime << "- Checking matrix info: doZ: " << info_.doZ << "\n";
  }

  void DualTimeMatrixManager::setup_assemblers( std::map<Expr::Tag,int> rhsIdxMap,
                                                std::map<Expr::Tag,int> varIdxMap,
                                                std::map<Expr::Tag,Expr::Tag> varRhsMap )
  {

    dbg_dualtime << "\n\n\nMatrix Manager\n";
    dbg_dualtime << "RHS-IDX map:\n";
    for( const auto& x : rhsIdxMap ){
      dbg_dualtime << "  " << x.first << " -> " << x.second << '\n';
    }
    dbg_dualtime << '\n';
    dbg_dualtime << "VAR-IDX map:\n";
    for( const auto& x : varIdxMap ){
      dbg_dualtime << "  " << x.first << " -> " << x.second << '\n';
    }
    dbg_dualtime << '\n';
    dbg_dualtime << "VAR-RHS map:\n";
    for( const auto& x : varRhsMap ){
      dbg_dualtime << "  " << x.first << " -> " << x.second << '\n';
    }
    dbg_dualtime << '\n';

    if( info_.doCompressible ){
      const Expr::matrix::OrdinalType nSpecies = info_.doSpecies ? info_.speciesDensities.size() : 1;
      Expr::TagList primitiveVariables;
      primitiveVariables.push_back( info_.density );
      primitiveVariables.push_back( info_.temperature );
      if( info_.doX ) primitiveVariables.push_back( info_.xVelocity );
      if( info_.doY ) primitiveVariables.push_back( info_.yVelocity );
      if( info_.doZ ) primitiveVariables.push_back( info_.zVelocity );
      if( info_.doSpecies ){
        for( int i=0; i<nSpecies - 1; ++i ){
          primitiveVariables.push_back( info_.massFractions[i] );
        }
      }
      dbg_dualtime << "PRIMITIVES:\n";
      for( const auto& p : primitiveVariables ){
        dbg_dualtime << p << '\n';
      }
      dbg_dualtime << '\n';

      std::map<Expr::Tag,Expr::Tag> primConsMap;
      primConsMap[info_.density] = info_.density;
      primConsMap[info_.temperature] = info_.totalEnergy;
      if( info_.doX ) primConsMap[info_.xVelocity] = info_.xMomentum;
      if( info_.doY ) primConsMap[info_.yVelocity] = info_.yMomentum;
      if( info_.doZ ) primConsMap[info_.zVelocity] = info_.zMomentum;
      if( info_.doSpecies ){
        for( int i=0; i<nSpecies - 1; ++i ){
          primConsMap[info_.massFractions[i]] = info_.speciesDensities[i];
        }
      }
      dbg_dualtime << "PRIM-CONS map:\n";
      for( const auto& x : primConsMap ){
        dbg_dualtime << "  " << x.first << " -> " << x.second << '\n';
      }
      dbg_dualtime << '\n';

      dbg_dualtime << "NSPECIES = " << nSpecies << '\n';
      dbg_dualtime << '\n';


      const Expr::matrix::OrdinalType xMomIdx = varIdxMap[info_.xMomentum];
      const Expr::matrix::OrdinalType yMomIdx = varIdxMap[info_.yMomentum];
      const Expr::matrix::OrdinalType zMomIdx = varIdxMap[info_.zMomentum];
      const Expr::matrix::OrdinalType egyIdx  = varIdxMap[info_.totalEnergy];
      const Expr::matrix::OrdinalType rhoConsIdx = varIdxMap[info_.density];
      std::vector<Expr::matrix::OrdinalType> speciesDensityIndices;
      for( int i=0; i<nSpecies - 1; ++i ){
        speciesDensityIndices.push_back( varIdxMap[info_.speciesDensities[i]] );
      }

      dbg_dualtime << "INDICES:\n";
      dbg_dualtime << "rho  idx: " << rhoConsIdx << '\n';
      dbg_dualtime << "egy  idx: " << egyIdx << '\n';
      dbg_dualtime << "xmom idx: " << xMomIdx << '\n';
      dbg_dualtime << "ymom idx: " << yMomIdx << '\n';
      dbg_dualtime << "zmom idx: " << zMomIdx << '\n';
      for( int i=0; i<nSpecies - 1; ++i ){
        dbg_dualtime << "species " << info_.massFractions[i] << " idx: " << speciesDensityIndices[i] << '\n';
      }
      dbg_dualtime << '\n';




      jacInviscid_ = boost::make_shared<InviscidFluxJacT>( "inviscid flux Jacobian",
                                                           info_.universalGasConstant,
                                                           info_.molecularWeights,
                                                           info_.density,
                                                           info_.cpHeatCapacity,
                                                           info_.totalEnthalpy,
                                                           info_.xCoord,
                                                           info_.yCoord,
                                                           info_.zCoord,
                                                           info_.doX ? info_.xVelocity : Expr::Tag(),
                                                           info_.doY ? info_.yVelocity : Expr::Tag(),
                                                           info_.doZ ? info_.zVelocity : Expr::Tag(),
                                                           info_.temperature,
                                                           info_.pressure,
                                                           info_.enthalpies,
                                                           info_.massFractions,
                                                           rhoConsIdx,
                                                           rhoConsIdx,
                                                           egyIdx,
                                                           egyIdx,
                                                           xMomIdx,
                                                           yMomIdx,
                                                           zMomIdx,
                                                           xMomIdx,
                                                           yMomIdx,
                                                           zMomIdx,
                                                           speciesDensityIndices,
                                                           speciesDensityIndices );

      jacViscous_ = boost::make_shared<ViscousFluxJacT> ( "viscous flux Jacobian",
                                                          info_.density,
                                                          info_.viscosity,
                                                          info_.conductivity,
                                                          info_.xCoord,
                                                          info_.yCoord,
                                                          info_.zCoord,
                                                          info_.doX ? info_.xVelocity : Expr::Tag(),
                                                          info_.doY ? info_.yVelocity : Expr::Tag(),
                                                          info_.doZ ? info_.zVelocity : Expr::Tag(),
                                                          info_.doSpecies ? info_.enthalpies : Expr::TagList(),
                                                          info_.doSpecies ? info_.diffusivities : Expr::TagList(),
                                                          egyIdx,
                                                          egyIdx,
                                                          xMomIdx,
                                                          yMomIdx,
                                                          zMomIdx,
                                                          xMomIdx,
                                                          yMomIdx,
                                                          zMomIdx,
                                                          speciesDensityIndices,
                                                          speciesDensityIndices );

      preconditioner_ = boost::make_shared<PreconditionerT> ( "acoustic preconditioner",
                                                              info_.universalGasConstant,
                                                              info_.molecularWeights,
                                                              info_.density,
                                                              info_.cvHeatCapacity,
                                                              info_.cpHeatCapacity,
                                                              info_.viscosity,
                                                              info_.conductivity,
                                                              info_.totalEnergy,
                                                              info_.xCoord,
                                                              info_.yCoord,
                                                              info_.zCoord,
                                                              info_.doX ? info_.xVelocity : Expr::Tag(),
                                                              info_.doY ? info_.yVelocity : Expr::Tag(),
                                                              info_.doZ ? info_.zVelocity : Expr::Tag(),
                                                              info_.soundSpeed,
                                                              info_.temperature,
                                                              info_.pressure,
                                                              info_.mmw,
                                                              info_.doSpecies ? info_.energies : Expr::TagList(),
                                                              info_.doSpecies ? info_.massFractions : Expr::TagList(),
                                                              info_.doSpecies ? info_.diffusivities : Expr::TagList(),
                                                              rhoConsIdx,
                                                              rhoConsIdx,
                                                              egyIdx,
                                                              egyIdx,
                                                              xMomIdx,
                                                              yMomIdx,
                                                              zMomIdx,
                                                              xMomIdx,
                                                              yMomIdx,
                                                              zMomIdx,
                                                              speciesDensityIndices,
                                                              speciesDensityIndices );

      stateTransformJ_ = boost::make_shared<StateTransformT> ( "state transformation for Jacobian",
                                                               info_.density,
                                                               info_.cvHeatCapacity,
                                                               info_.totalEnergy,
                                                               info_.doX ? info_.xVelocity : Expr::Tag(),
                                                               info_.doY ? info_.yVelocity : Expr::Tag(),
                                                               info_.doZ ? info_.zVelocity : Expr::Tag(),
                                                               info_.doSpecies ? info_.energies : Expr::TagList(),
                                                               info_.doSpecies ? info_.massFractions : Expr::TagList(),
                                                               rhoConsIdx,
                                                               rhoConsIdx,
                                                               egyIdx,
                                                               egyIdx,
                                                               xMomIdx,
                                                               yMomIdx,
                                                               zMomIdx,
                                                               xMomIdx,
                                                               yMomIdx,
                                                               zMomIdx,
                                                               speciesDensityIndices,
                                                               speciesDensityIndices );
      stateTransformP_ = boost::make_shared<StateTransformT> ( "state transformation for Preconditioner",
                                                               info_.density,
                                                               info_.cvHeatCapacity,
                                                               info_.totalEnergy,
                                                               info_.doX ? info_.xVelocity : Expr::Tag(),
                                                               info_.doY ? info_.yVelocity : Expr::Tag(),
                                                               info_.doZ ? info_.zVelocity : Expr::Tag(),
                                                               info_.doSpecies ? info_.energies : Expr::TagList(),
                                                               info_.doSpecies ? info_.massFractions : Expr::TagList(),
                                                               rhoConsIdx,
                                                               rhoConsIdx,
                                                               egyIdx,
                                                               egyIdx,
                                                               xMomIdx,
                                                               yMomIdx,
                                                               zMomIdx,
                                                               xMomIdx,
                                                               yMomIdx,
                                                               zMomIdx,
                                                               speciesDensityIndices,
                                                               speciesDensityIndices );

      if( info_.doSpecies ){
        for( int i=0; i<nSpecies - 1; ++i ){
          const Expr::matrix::OrdinalType rhsIdx = rhsIdxMap[varRhsMap[info_.speciesDensities[i]]];
          const Expr::Tag prodRate = info_.productionRates[i];
          for( const auto& primTag : primitiveVariables ){
            if( primTag != info_.xVelocity && primTag != info_.yVelocity && primTag != info_.zVelocity ){
              const Expr::matrix::OrdinalType varIdx = varIdxMap[primConsMap[primTag]];
              jacChemistry_->element<FieldT>( rhsIdx, varIdx ) = Expr::matrix::sensitivity( prodRate, primTag );
            }
          }
        }
      }
      jacChemistry_->finalize();
    }
    else{ /* else if info_.doCompressible */
      for( const auto& rhsTag : info_.scalarRightHandSides ){
        for( const auto& varTag : info_.scalarVariables ){
          jacScalars_->element<FieldT>( rhsIdxMap[rhsTag], varIdxMap[varTag] ) = Expr::matrix::sensitivity( rhsTag, varTag );
        }
      }
      jacScalars_->finalize();
    }
  }

  void DualTimeMatrixManager::setup_matrix_assembly( BlockImplicitBDF* dtIntegrator ){

    // get maps for matrix construction
    std::map<Expr::Tag,int> varIdxMap = dtIntegrator->variable_tag_index_map( Expr::STATE_DYNAMIC );
    std::map<Expr::Tag,int> rhsIdxMap = dtIntegrator->rhs_tag_index_map();
    std::map<Expr::Tag,Expr::Tag> varRhsMap = dtIntegrator->variable_rhs_tag_map( Expr::STATE_DYNAMIC );

    // set up the assemblers and set the Jacobian and preconditioner
    print_tags();
    setup_assemblers( rhsIdxMap, varIdxMap, varRhsMap );

    if( info_.doCompressible ){
      if( info_.doImplicitInviscid ){
        if( info_.doSpecies ){
          if( info_.doPreconditioning ){
            // compressible + multispecies + inviscid + preconditioning
            dtIntegrator->set_jacobian_and_preconditioner( ( jacViscous_ - jacInviscid_ + jacChemistry_ ) * stateTransformJ_,
                                                           preconditioner_ * stateTransformP_ );
          }
          else{
            // compressible + multispecies + inviscid
            dtIntegrator->set_jacobian( ( jacViscous_ - jacInviscid_ + jacChemistry_ ) * stateTransformJ_ );
          }
        }
        else{
          if( info_.doPreconditioning ){
            // compressible + inviscid + preconditioning
            dtIntegrator->set_jacobian_and_preconditioner( ( jacViscous_ - jacInviscid_ ) * stateTransformJ_,
                                                           preconditioner_ * stateTransformP_ );
          }
          else{
            // compressible + inviscid
            dtIntegrator->set_jacobian( ( jacViscous_ - jacInviscid_ ) * stateTransformJ_ );
          }
        }
      }
      else{
        if( info_.doSpecies ){
          if( info_.doPreconditioning ){
            // compressible + multispecies + preconditioning
            dtIntegrator->set_jacobian_and_preconditioner( ( jacViscous_ + jacChemistry_ ) * stateTransformJ_,
                                                           preconditioner_ * stateTransformP_ );
          }
          else{
            // compressible + multispecies
            dtIntegrator->set_jacobian( ( jacViscous_ + jacChemistry_ ) * stateTransformJ_ );
          }
        }
        else{
          if( info_.doPreconditioning ){
            // compressible + preconditioning
            dtIntegrator->set_jacobian_and_preconditioner( jacViscous_ * stateTransformJ_,
                                                           preconditioner_ * stateTransformP_ );
          }
          else{
            //compressible
            boost::shared_ptr<Expr::matrix::SparseMatrix<SVolField> > zero = boost::make_shared<Expr::matrix::SparseMatrix<SVolField> >();
            zero->finalize();
//            dtIntegrator->set_jacobian( jacViscous_ * stateTransformJ_ );
            dtIntegrator->set_jacobian( zero );
          }
        }
      }

//          // begin matrix debugging  /* please keep this code here so nobody else has to figure it out. this is a demo of how one can print an assembled matrix if desperate */
//          boost::shared_ptr<Expr::matrix::SparseMatrix<SVolField> > tester = boost::make_shared<Expr::matrix::SparseMatrix<SVolField> >();
//          tester->element<SVolField>(0, 0) = info_.temperature;
//          tester->element<SVolField>(0, 1) = info_.pressure;
//          tester->element<SVolField>(0, 2) = info_.density;
//          tester->element<SVolField>(0, 3) = info_.xVelocity;
//          tester->element<SVolField>(1, 0) = Expr::matrix::sensitivity( Expr::Tag( "rr_H2O2", Expr::STATE_NONE ), Expr::Tag( "T", Expr::STATE_NONE ) );
//          tester->finalize();
//          const Expr::TagList matrixDebugTags = Expr::matrix::matrix_tags( "matrix-debug", 11 );
//          using MatrixAssemblerExprType = Expr::matrix::MatrixExpression<SVolField>::Builder;
//          dtIntegrator->register_root_expression( new MatrixAssemblerExprType( matrixDebugTags, tester, true ) );
//          // end matrix debugging

    }
    else{
      // solving scalar transport equations with constant density, passive velocity field
      dtIntegrator->set_jacobian( jacScalars_ );
    }
  }

} // namespace WasatchCore
