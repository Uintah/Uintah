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

#ifndef Wasatch_MatrixAssembly_h
#define Wasatch_MatrixAssembly_h

#include <spatialops/structured/FVStaggered.h>

#include <expression/matrix-assembly/DenseSubMatrix.h>
#include <expression/matrix-assembly/ScaledIdentityMatrix.h>
#include <expression/matrix-assembly/compressible-reactive-flow/ViscousFluxAssembler.h>
#include <expression/matrix-assembly/compressible-reactive-flow/InviscidFluxAssembler.h>
#include <expression/matrix-assembly/compressible-reactive-flow/StateTransformAssembler.h>
#include <expression/matrix-assembly/compressible-reactive-flow/AcousticPreconditioner.h>
#include <expression/matrix-assembly/Compounds.h>
#include <expression/matrix-assembly/MapUtilities.h>


namespace Expr{
  namespace DualTime{
    template< typename T1, typename T2 > class BlockImplicitBDFDualTimeIntegrator;
  }
}


namespace WasatchCore{

  struct DualTimeMatrixInfo
  {
    // the tag/quantity                    where this tag/quantity is set
    Expr::Tag viscosity,                   // momentum equations
              density,                     // momentum equations
              cpHeatCapacity,              // energy equation
              cvHeatCapacity,              // energy equation
              conductivity,                // energy equation
              pressure,                    // momentum equations
              temperature,                 // momentum equations
              totalEnergy,                 // energy equation
              totalEnthalpy,               // energy equation
              xCoord,                      // Wasatch.cc
              yCoord,                      // Wasatch.cc
              zCoord,                      // Wasatch.cc
              xMomentum,                   // momentum equations
              yMomentum,                   // momentum equations
              zMomentum,                   // momentum equations
              xVelocity,                   // momentum equations
              yVelocity,                   // momentum equations
              zVelocity,                   // momentum equations
              soundSpeed,                  // Wasatch.cc
              mmw,                         // species equations, energy equation
              timeStepSize;                // Wasatch.cc
    Expr::TagList massFractions,           // species equations
                  speciesDensities,        // species equations
                  enthalpies,              // energy equation
                  energies,                // energy equation
                  diffusivities,           // species equations
                  productionRates,         // species equations
                  scalarRightHandSides,    // scalar equations
                  scalarVariables;         // scalar equations

    bool doSpecies = false;                // species equations
    bool doX = false;                      // momentum equations
    bool doY = false;                      // momentum equations
    bool doZ = false;                      // momentum equations
    bool doCompressible = false;           // momentum equations

    bool doImplicitInviscid = true;        // Wasatch.cc
    bool doPreconditioning = true;         // Wasatch.cc
    bool doBlockImplicit = false;          // Wasatch.cc

    bool doLocalCflVnn = false;            // Wasatch.cc
    double cfl = 1.0;                      // Wasatch.cc
    double vnn = 1.0;                      // Wasatch.cc

    double dsMax = 1.e6;                   // Wasatch.cc
    double dsMin = 1.e-9;                  // Wasatch.cc

    double constantDs;                     // Wasatch.cc

    int logIterationRate = 100000;         // Wasatch.cc
    int maxIterations = 100;               // Wasatch.cc
    double tolerance = 1.e-8;              // Wasatch.cc


    double universalGasConstant;           // energy equation
    std::vector<double> molecularWeights;  // species equations


    void set_mass_fraction_tags( const Expr::TagList tags );

    void set_species_density_tags( const Expr::TagList tags );

    void set_enthalpies( const Expr::TagList tags );

    void set_energies( const Expr::TagList tags );

    void set_diffusivities( const Expr::TagList tags );

    void set_production_rates( const Expr::TagList tags );

    void set_molecular_weights( const std::vector<double> mw );

    void add_scalar_equation( const Expr::Tag solution_variable_tag, const Expr::Tag rhs_tag );


  };

  using FieldT           = SpatialOps::SVolField;
  using InviscidFluxJacT = Expr::matrix::InviscidFluxAssembler<FieldT>;
  using ViscousFluxJacT  = Expr::matrix::ViscousFluxAssembler<FieldT>;
  using StateTransformT  = Expr::matrix::StateTransformAssembler<FieldT>;
  using PreconditionerT  = Expr::matrix::AcousticPreconditionerAssembler<FieldT>;
  using DenseSubMatrixT  = Expr::matrix::DenseSubMatrix<FieldT>;
  using BlockImplicitBDF = Expr::DualTime::BlockImplicitBDFDualTimeIntegrator<SVolField, SVolField>;

  class DualTimeMatrixManager
  {

    DualTimeMatrixInfo info_;

    boost::shared_ptr<InviscidFluxJacT> jacInviscid_;
    boost::shared_ptr<ViscousFluxJacT>  jacViscous_;
    boost::shared_ptr<PreconditionerT>  preconditioner_;
    boost::shared_ptr<StateTransformT>  stateTransformJ_;
    boost::shared_ptr<StateTransformT>  stateTransformP_;


    boost::shared_ptr<DenseSubMatrixT>  jacChemistry_ = boost::make_shared<DenseSubMatrixT> ( "chemical source Jacobian" );
    boost::shared_ptr<DenseSubMatrixT>  jacScalars_   = boost::make_shared<DenseSubMatrixT> ( "scalar RHS Jacobian" );

    void print_tags();

    void setup_assemblers( std::map<Expr::Tag,int> rhsIdxMap,
                           std::map<Expr::Tag,int> varIdxMap,
                           std::map<Expr::Tag,Expr::Tag> varRhsMap );

    void setup_matrix_assembly( BlockImplicitBDF* dtIntegrator );

  public:
    DualTimeMatrixManager( BlockImplicitBDF* dtIntegrator, const DualTimeMatrixInfo& info )
      : info_( info )
    {
      setup_matrix_assembly( dtIntegrator );
    }

  };

} // namespace WasatchCore

#endif // Wasatch_MatrixAssembly_h
