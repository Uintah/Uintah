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

#include <CCA/Components/Wasatch/Expressions/DORadSolver.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
//#include <Core/Parallel/Parallel.h>
//#include <CCA/Ports/LoadBalancer.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>

//==============================================================================

OrdinateDirections::SVec::SVec( const double _x,
                                const double _y,
                                const double _z,
                                const double _w )
: x(_x), y(_y), z(_z), w(_w)
{}

//-----------------------------------------------------------------

void
assign_variants( std::vector<OrdinateDirections::SVec>& vec,
                 const OrdinateDirections::SVec& s )
{
  const double pi = 3.141592653589793;
  vec.push_back( OrdinateDirections::SVec(  s.x,  s.y,  s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec( -s.x,  s.y,  s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec(  s.x, -s.y,  s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec( -s.x, -s.y,  s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec(  s.x,  s.y, -s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec( -s.x,  s.y, -s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec(  s.x, -s.y, -s.z, s.w/4.0*pi ) );
  vec.push_back( OrdinateDirections::SVec( -s.x, -s.y, -s.z, s.w/4.0*pi ) );
}

//-----------------------------------------------------------------

OrdinateDirections::OrdinateDirections( const int n )
{
  // These ordinate values are from Modest's radiative heat transfer book,
  // page 545, and the quadrature weights have a factor of 4*pi in them.
  // We remove that factor of 4*pi in the assign_variants function.
  switch(n){
    case 2:
      assign_variants( ordinates_, SVec( 0.5773503, 0.5773503, 0.5773503, 1.5707963 ) );
      break;
    case 4:
      assign_variants( ordinates_, SVec( 0.2958759, 0.2958759, 0.9082483, 0.5235987 ) );
      assign_variants( ordinates_, SVec( 0.2958759, 0.9082483, 0.2958759, 0.5235987 ) );
      assign_variants( ordinates_, SVec( 0.9082483, 0.2958759, 0.2958759, 0.5235987 ) );
      break;
    case 6:
      assign_variants( ordinates_, SVec( 0.1838670, 0.1838670, 0.9656013, 0.1609517 ) );
      assign_variants( ordinates_, SVec( 0.1838670, 0.6959514, 0.6959514, 0.3626469 ) );
      assign_variants( ordinates_, SVec( 0.1838670, 0.9656013, 0.1838670, 0.1609517 ) );
      assign_variants( ordinates_, SVec( 0.6959514, 0.1838670, 0.6959514, 0.3626469 ) );
      assign_variants( ordinates_, SVec( 0.6959514, 0.6959514, 0.1838670, 0.3626469 ) );
      assign_variants( ordinates_, SVec( 0.9656013, 0.1838670, 0.1838670, 0.1609517 ) );
      break;
    case 8:
      assign_variants( ordinates_, SVec( 0.1422555, 0.1422555, 0.9795543, 0.1712359 ) );
      assign_variants( ordinates_, SVec( 0.1422555, 0.5773503, 0.8040087, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.1422555, 0.8040087, 0.5773503, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.1422555, 0.9795543, 0.1422555, 0.1712359 ) );
      assign_variants( ordinates_, SVec( 0.5773503, 0.1422555, 0.8040087, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.5773503, 0.5773503, 0.5773503, 0.4617179 ) );
      assign_variants( ordinates_, SVec( 0.5773503, 0.8040087, 0.1422555, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.8040087, 0.1422555, 0.5773503, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.8040087, 0.5773503, 0.1422555, 0.0992284 ) );
      assign_variants( ordinates_, SVec( 0.9795543, 0.1422555, 0.1422555, 0.1712359 ) );
      break;
    default:
      throw std::invalid_argument("Invalid order for discrete ordinates. Must be 2, 4, 6 or 8\n" ); ;
  }

# ifndef NDEBUG
  double xsum=0, ysum=0, zsum=0, wsum=0.0;
  BOOST_FOREACH( const SVec& v, ordinates_ ){
    xsum += v.x;
    ysum += v.y;
    zsum += v.z;
    wsum += v.w;
  }
  assert( std::abs( xsum ) < 1e-12 );
  assert( std::abs( ysum ) < 1e-12 );
  assert( std::abs( zsum ) < 1e-12 );
  assert( std::abs( wsum ) < 1e-12 );
# endif
}

//-----------------------------------------------------------------

OrdinateDirections::~OrdinateDirections()
{}

//==============================================================================

namespace WasatchCore {

  Expr::TagList DORadSolver::intensityTags = Expr::TagList();

  DORadSolver::DORadSolver( const std::string intensityName,
                            const std::string intensityRHSName,
                            const OrdinateDirections::SVec& svec,
                            const Expr::Tag& absCoefTag,
                            const Expr::Tag& scatCoefTag,
                            const Expr::Tag& temperatureTag,
                            Uintah::SolverInterface& solver )
  : Expr::Expression<SVolField>(),

    svec_( svec ),
    hasAbsCoef_ (  absCoefTag != Expr::Tag() ),
    hasScatCoef_( scatCoefTag != Expr::Tag() ),
  
    temperatureTag_(temperatureTag),

    doX_( true ),
    doY_( true ),
    doZ_( true ),

    materialID_( 0 ),

    solver_( solver ),

    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_   ( Uintah::VarLabel::create( intensityName + "_matrix",
                                               Uintah::CCVariable<Uintah::Stencil7>::getTypeDescription() ) ),
    intensityLabel_( Uintah::VarLabel::create( intensityName,
                                               WasatchCore::get_uintah_field_type_descriptor<SVolField>() ) ),
    rhsLabel_      ( Uintah::VarLabel::create( intensityRHSName,
                                               WasatchCore::get_uintah_field_type_descriptor<SVolField>() ) )
  {
    this->set_gpu_runnable( false );
    
     temperature_ = create_field_request<SVolField>(temperatureTag);
    if( hasAbsCoef_  )   absCoef_ = create_field_request<SVolField>(absCoefTag);
    if( hasScatCoef_ )  scatCoef_ = create_field_request<SVolField>(scatCoefTag);

  }

  //--------------------------------------------------------------------

  DORadSolver::~DORadSolver()
  {
    Uintah::VarLabel::destroy( matrixLabel_    );
    Uintah::VarLabel::destroy( intensityLabel_ );
    Uintah::VarLabel::destroy( rhsLabel_       );
  }

  //--------------------------------------------------------------------

  void
  DORadSolver::schedule_solver( const Uintah::LevelP& level,
                                Uintah::SchedulerP sched,
                                const Uintah::MaterialSet* const materials,
                                const int rkStage,
                                const bool isDoingInitialization )
  {
    if( rkStage != 1 ) return;
    solver_.scheduleSolve( level, sched, materials, matrixLabel_,
        Uintah::Task::NewDW,
                           intensityLabel_,
                           true,
                           rhsLabel_, Uintah::Task::NewDW,
                           intensityLabel_, Uintah::Task::NewDW,
                           (rkStage == 1) );
  }

  //--------------------------------------------------------------------

  void
  DORadSolver::declare_uintah_vars( Uintah::Task& task,
                                    const Uintah::PatchSubset* const patches,
                                    const Uintah::MaterialSubset* const materials,
                                    const int rkStage )
  {
    if( rkStage != 1 ) return;
    task.computes( matrixLabel_, patches, Uintah::Task::ThisLevel, materials, Uintah::Task::NormalDomain );
  }

  //--------------------------------------------------------------------

  void
  DORadSolver::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                                 const Uintah::Patch* const patch,
                                 const int material,
                                 const int rkStage )
  {
    materialID_ = material;
    patch_ = const_cast<Uintah::Patch*>( patch );
    rkStage_ = rkStage;
    if( rkStage_ != 1 ) return;

    dw->allocateAndPut( matrix_, matrixLabel_, materialID_, patch );
  }

  //--------------------------------------------------------------------

  void DORadSolver::setup_matrix( SVolField& rhs, const SVolField& temperature )
  {
    std::cout << "DORadSolver::setup_matrix() for " << this->get_tags() << "\n";
    const Uintah::Vector spacing = patch_->dCell();

    const double dx = spacing[0];
    const double dy = spacing[1];
    const double dz = spacing[2];

    const OrdinateDirections::SVec& sn = svec_;

    // p is current cell
    double p = 0.0;
    // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
    double n=0.0, s=0.0, e=0.0, w=0.0, t=0.0, b=0.0;

    // calculate the entries in the coefficient matrix when no BC is present
    if( doX_ ){
      if( sn.x > 0 ){ e = -sn.x/dx; p -= e; }
      else          { w =  sn.x/dx; p -= w; }
    }
    if( doY_ ){
      if( sn.y > 0 ){ n = -sn.y/dy; p -= n; }
      else          { s =  sn.y/dy; p -= s; }
    }
    if( doZ_ ){
      if( sn.z > 0 ){ t = -sn.z/dz; p -= t; }
      else          { b = sn.z/dz;  p -= b; }
    }

    for( Uintah::CellIterator iter(patch_->getCellIterator()); !iter.done(); iter++ ){

      const Uintah::IntVector iCell = *iter;
      Uintah::Stencil7& coefs = matrix_[iCell];

      // jcs the absCoef_ and scatCoef_ fields are SpatialFields. We need to index them appropriately.
      const SpatialOps::IntVec index( iCell[0], iCell[1], iCell[2] );
      p -= hasAbsCoef_ ? (absCoef_->field_ref())(index) : 1.0;

      if( hasScatCoef_ ) p += (scatCoef_->field_ref())(index);

      // This only applies when a boundary is not present.
      // For boundaries, we set the intensity directly.
      coefs.n = n;
      coefs.s = s;
      coefs.e = e;
      coefs.w = w;
      coefs.t = t;
      coefs.b = b;
      coefs.p = p;

      // jcs need to set boundary conditions...
    }

    // When boundary conditions are present, modify the coefficient matrix coefficients at the boundary
    if( patch_->hasBoundaryFaces() ){
      /* ALGORITHM:
         1. loop over the patches
         2. For each patch, loop over materials
         3. For each material, loop over boundary faces
         4. For each boundary face, loop over its children
         5. For each child, get the cell faces and set appropriate boundary conditions
       */
      using SpatialOps::IntVec;
      const Uintah::IntVector uintahPatchDim = patch_->getCellHighIndex();
      const IntVec patchDim( uintahPatchDim[0], uintahPatchDim[1], uintahPatchDim[2] );

      std::vector<Uintah::Patch::FaceType> bndFaces;
      patch_->getBoundaryFaces( bndFaces );

      // loop over the boundary faces
      BOOST_FOREACH( const Uintah::Patch::FaceType face, bndFaces ){

        //get the number of children
        const int numChildren = patch_->getBCDataArray(face)->getNumberChildren( materialID_ );

        for( int child=0; child<numChildren; ++child ){

          Uintah::Iterator boundPtr;
          Uintah::Iterator nu;
//          const Uintah::BoundCondBase* const bc = patch_->getArrayBCValues( face, materialID_, temperatureTag_.name(), boundPtr, nu, child );

//          const bool hasExtraCells = ( patch_->getExtraCells() != Uintah::IntVector(0,0,0) );

          // cell offset used to calculate local cell index with respect to patch.
          const Uintah::IntVector patchCellOffset = patch_->getCellLowIndex(0);

          for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {

            // jcs note that boundaries are faces, not cells. Perhaps we should
            // be taking that into account properly...?  This would require changes
            // to the coefficient values in the matrix and would also require
            // a bit more work on the RHS to determine the appropriate temperature.
            Uintah::IntVector bcPointIndex(*boundPtr);
            Uintah::Stencil7& coefs = matrix_[bcPointIndex];
            coefs.n = 0.0;  coefs.s = 0.0;
            coefs.e = 0.0;  coefs.w = 0.0;
            coefs.t = 0.0;  coefs.b = 0.0;
            coefs.p = 1.0;

            SpatialOps::IntVec soIndex( bcPointIndex[0], bcPointIndex[1], bcPointIndex[2] );

            const double sigma = 5.67037321e-8; // Stefan-Boltzmann constant, W/(m^2 K^4)
            const double pi = 3.141592653589793;
            const double t = temperature(soIndex);
            const double t2 = t*t;
            const double t4 = t2*t2;
            const double abscoef = hasAbsCoef_ ? (absCoef_->field_ref())(soIndex) : 1.0;

            // rhs is the black body intensity
            rhs(soIndex) = sigma * abscoef * t4 / pi;

          }
        } // child loop
      } // face loop
    }
    std::cout << "\tDONE SETTING UP DOM SYSTEM\n";
  }

  //--------------------------------------------------------------------

  void DORadSolver::evaluate()
  {
    using namespace SpatialOps;
    typedef typename Expr::Expression<SVolField>::ValVec SVolFieldVec;

    SVolFieldVec& results = this->get_value_vec();
    SVolField& intensity = *results[0];
    SVolField& rhs       = *results[1];

    const double sigma = 5.67037321e-8; // Stefan-Boltzmann constant, W/(m^2 K^4)
    const double pi = 3.141592653589793;
    const SVolField& temperature = temperature_->field_ref();
    if( hasAbsCoef_ ) rhs <<= absCoef_->field_ref() * sigma * pow(temperature,4) / pi;
    else              rhs <<=             sigma * pow(temperature,4) / pi;

    setup_matrix( rhs, temperature );

    // jcs set guess to be sigma * T^4
    intensity <<= sigma * pow( temperature, 4 );

    // the linear system is solved after this...
  }

  //--------------------------------------------------------------------

  DORadSolver::Builder::Builder( const std::string intensityName,
                                 const OrdinateDirections::SVec& svec,
                                 const Expr::Tag absCoefTag,
                                 const Expr::Tag scatCoefTag,
                                 const Expr::Tag temperatureTag,
                                 Uintah::SolverInterface& solver )
  : Expr::ExpressionBuilder( Expr::tag_list( Expr::Tag( intensityName,          Expr::STATE_NONE ),
                                             Expr::Tag( intensityName + "_RHS", Expr::STATE_NONE ) ) ),
    absCoefTag_    ( absCoefTag     ),
    scatCoefTag_   ( scatCoefTag    ),
    temperatureTag_( temperatureTag ),
    solver_        ( solver         ),
    svec_          ( svec           )
  {
    DORadSolver::intensityTags.push_back( Expr::Tag( intensityName, Expr::STATE_NONE ) );
  }

  //--------------------------------------------------------------------

  Expr::ExpressionBase* DORadSolver::Builder::build() const
  {
    const Expr::TagList& tags = this->get_tags();
    return new DORadSolver( tags[0].name(), tags[1].name(), svec_, absCoefTag_, scatCoefTag_, temperatureTag_, solver_ );
  }

  //============================================================================

  DORadSrc::
  DORadSrc( const Expr::Tag& temperatureTag,
            const Expr::Tag& absCoefTag,
            const OrdinateDirections& ord )
    : Expr::Expression<SVolField>(),
      ord_           ( ord            ),
      hasAbsCoef_    ( absCoefTag != Expr::Tag() )
  {
    
    create_field_vector_request<SVolField>(DORadSolver::intensityTags, intensity_);
     temperature_ = create_field_request<SVolField>(temperatureTag);
    if (hasAbsCoef_)  absCoef_ = create_field_request<SVolField>(absCoefTag);
  }

  //--------------------------------------------------------------------

  DORadSrc::~DORadSrc(){}

  //--------------------------------------------------------------------

  void DORadSrc::evaluate()
  {
    using namespace SpatialOps;
    SVolField& divQ = this->value();

    const SVolField& temperature = temperature_->field_ref();
    // First, store the scalar flux in divQ.  The scalar flux is:
    //  G = \sum \omega_\ell I^{(\ell)}
    divQ <<= intensity_[0]->field_ref() * ord_.get_ordinate_information(0).w;
    for( size_t i=1; i<intensity_.size(); ++i ){
      divQ <<= divQ + intensity_[i]->field_ref() * ord_.get_ordinate_information(i).w;
    }

    // now calculate div Q:  divQ = \kappa ( 4\sigma T^4 - G )
    const double sigma = 5.67037321e-8; // Stefan-Boltzmann constant, W/(m^2 K^4)
    if( hasAbsCoef_ ) divQ <<= absCoef_->field_ref() * ( 4 * sigma * pow(temperature,4) - divQ );
    else              divQ <<=             ( 4 * sigma * pow(temperature,4) - divQ );
  }

  //--------------------------------------------------------------------

  DORadSrc::
  Builder::Builder( const Expr::Tag divQTag,
                    const Expr::Tag temperatureTag,
                    const Expr::Tag absCoefTag,
                    const OrdinateDirections& ord )
    : ExpressionBuilder( divQTag ),
      temperatureTag_( temperatureTag ),
      absCoefTag_    ( absCoefTag     ),
      ord_( ord )
  {
    assert( ord.number_of_directions() == DORadSolver::intensityTags.size() );
  }

  //--------------------------------------------------------------------

  Expr::ExpressionBase* DORadSrc::Builder::build() const{
    return new DORadSrc( temperatureTag_, absCoefTag_, ord_ );
  }

  //============================================================================

} // namespace WasatchCore
