#ifndef ParticleInitialization_h
#define ParticleInitialization_h

#include <expression/ExprLib.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>

// note that here we break with convention to allow uintah intrusion into an
// expression.  This is because these initialization expressions are fairly
// specific to Uintah/Wasatch.
#include <Core/Grid/Box.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <boost/random/uniform_int.hpp>
//--------------------------------------------------------------------

/**
 *  \brief Returns the lowest coordinate of a patch.
 *  
 *  \param patch  A pointer to a Uintah::Patch.
 *
 *  \param coord  A string denoting which coordinate we want this function to return. Allowed
 options are "X", "Y", and "Z".
 */
double
get_patch_low(const Uintah::Patch* const patch, const std::string& coord)
{
  if     ( coord == "X" ) return patch->getBox().lower().x();
  else if( coord == "Y" ) return patch->getBox().lower().y();
  else if( coord == "Z" ) return patch->getBox().lower().z();
  assert( false ); // should never get here.
  return 0.0;
}

//--------------------------------------------------------------------

/**
 *  \brief Returns the highest coordinate of a patch.
 *
 *  \param patch  A pointer to a Uintah::Patch.
 *
 *  \param coord  A string denoting which coordinate we want this function to return. Allowed
 options are "X", "Y", and "Z".
 */
double
get_patch_high(const Uintah::Patch* const patch, const std::string& coord)
{
  if     ( coord == "X" ) return patch->getBox().upper().x();
  else if( coord == "Y" ) return patch->getBox().upper().y();
  else if( coord == "Z" ) return patch->getBox().upper().z();
  assert( false ); // should never get here
  return 0.0;
}

//--------------------------------------------------------------------

/**
 *  \brief Returns the cell size.
 *
 *  \param patch  A pointer to a Uintah::Patch.
 *
 *  \param coord  A string denoting which coordinate we want this function to return. Allowed
 options are "X", "Y", and "Z".
 */
double
get_cell_size(const Uintah::Patch* const patch, const std::string& coord)
{
  if     ( coord == "X" ) return patch->dCell().x();
  else if( coord == "Y" ) return patch->dCell().y();
  else if( coord == "Z" ) return patch->dCell().z();
  assert( false ); // should never get here
  return 0.0;
}

//--------------------------------------------------------------------

/**
 *  \brief Returns a cell's minus face position. This is cell position - cellsize/2.
 *
 *  \param patch  A pointer to a Uintah::Patch.
 *
 *  \param coord  A string denoting which coordinate we want this function to return. Allowed
 options are "X", "Y", and "Z".
 *
 *  \param cellPosition  A Uintah::Point denoting the cell position (cell center).
 */
double
get_cell_position_offset(const Uintah::Patch* const patch, const std::string& coord, const Uintah::Point& cellPosition)
{
  if     ( coord == "X" ) return cellPosition.x() - patch->dCell().x()/2.0;
  else if( coord == "Y" ) return cellPosition.y() - patch->dCell().y()/2.0;
  else if( coord == "Z" ) return cellPosition.z() - patch->dCell().z()/2.0;
  assert( false ); // should never get here
  return 0.0;
}

//==================================================================

/**
 *  \class   ParticleGeometryBased
 *  \author  Tony Saad
 *  \date    July, 2014
 *  \ingroup Expressions
 *
 *  \brief The ParticleGeometryBased expression allows users to fill arbitrary geometry shapes
 with particles.
 */
class ParticleGeometryBased : public Expr::Expression<ParticleField>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     *  \brief Build a ParticleGeometryBased expression.
     *
     *  \param resultTag Expr::Tag of the resulting expression.
     *
     *  \param coord     String denoting the coordinate direction computed by this expression. 
     Allowed options are "X", "Y", and "Z".Note that you cannot use this expression to initialize 
     scalar particle properties.
     *
     *  \param seed      Integer to seed the random number generator.
     *
     *  \param gomeObjects A vector of GeometryPiece pointers that contains a list of all geometery objects
     to be filled by this expression.
     */
    Builder( const Expr::Tag& resultTag,
             const std::string& coord,
             const int seed,
             std::vector<Uintah::GeometryPieceP> geomObjects )
    : ExpressionBuilder(resultTag),
      coord_(coord),
      seed_(seed),
      geomObjects_(geomObjects)
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleGeometryBased( coord_, seed_, geomObjects_ );
    }

  private:
    const std::string coord_;
    const int seed_;
    std::vector <Uintah::GeometryPieceP > geomObjects_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    patchContainer_ = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
  }

  void evaluate()
  {
    using namespace SpatialOps;
    using namespace Uintah;
    ParticleField& result = this->value();

    const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();

    // random number generator
    typedef boost::mt19937 base_generator_type; // mersenne twister
    // seed the random number generator based on the MPI rank
    const int pid =  patchContainer_->get_uintah_patch()->getID();
    base_generator_type generator((unsigned) ( (pid+1) * (1 + seed_) ));
    const double dx =  get_cell_size(patch, coord_); // dx, dy, or dz
    boost::uniform_real<> rand_dist(0,dx); // generate random numbers between 0 and dx. Then we offset those by the cell location
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);

    //________________________________________________
    // collect the grid points that live inside the geometry
    const std::vector<Uintah::Point>& insidePoints = GeometryPieceFactory::getInsidePoints(patch);

    ParticleField::iterator phiIter = result.begin();
    ParticleField::iterator phiIterEnd = result.end();

    if( insidePoints.size() > 0 && phiIter < phiIterEnd ){
      // random distribution to pick random points within the geometry shape
      base_generator_type generator2((unsigned) ( (pid+1) ));
      boost::uniform_int<> rand_dist_int(0, insidePoints.size() - 1);
      boost::variate_generator<base_generator_type&, boost::uniform_int<> > boost_rand_int(generator2, rand_dist_int);

      //________________________________________________
      // now iterate over the inside points and fill in the particles
      while( phiIter < phiIterEnd ){
        const int idx = boost_rand_int();
        Uintah::Point insidePoint = insidePoints[idx];
        const double offset = get_cell_position_offset(patch, coord_, insidePoint);
        *phiIter = boost_rand() + offset;
        ++phiIter;
      }
    }
  }

  
private:
  const std::string coord_;
  const int seed_;
  typedef std::vector <Uintah::GeometryPieceP > GeomValueMapT;  // temporary typedef map that stores boundary
  ParticleGeometryBased( const std::string& coord, const int seed, GeomValueMapT geomObjects )
  : Expr::Expression<ParticleField>(),
    coord_(coord),
    seed_(seed),
    geomObjects_(geomObjects)
  {}

  GeomValueMapT geomObjects_;
  WasatchCore::UintahPatchContainer* patchContainer_;

};

//--------------------------------------------------------------------

/**
 *  \class  ParticleRandomIC
 *  \ingroup WasatchParticles
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Generates a pseudo-random field to initialize particle positions (x, y, and z). The 
 random field generated is generated within two bounds, low and high, that either specified by
 the user, or determined from the patch logical coordinates.
 */
class ParticleRandomIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleRandomIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     *  \brief            Build a ParticleRandomIC expression
     *  \param resultTag  The Tag for the resulting expression that his class computes.
     *  \param coord      A string designating which particle coordinate this expression computes.
     *                    These can only be X, Y, or Z.
     *  \param lo         The lower bound used in the random number generator. This value is
     *                    superseded by the lower patch boundaries if usePatchBounds is set to true.
     *
     *  \param hi         The upper bound used in the random number generator. This value is
     *                    superseded by the upper patch boundary if the usePatchBounds is set to true.
     *
     *  \param seed       The seed for the random number generator. This is a required quantity.
     *
     *  \param usePatchBounds If true, then use the boundaries of the uintah patch on which this
     *                    expression is executing.
     */
    Builder( const Expr::Tag& resultTag,
             const std::string& coord,
             const double lo,
             const double hi,
             const int seed,
             const bool usePatchBounds )
    : ExpressionBuilder(resultTag),
      coord_(coord),
      lo_(lo),
      hi_(hi),
      seed_(seed),
      usePatchBounds_(usePatchBounds)
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new ParticleRandomIC(coord_,lo_, hi_, seed_,usePatchBounds_ );
    }

  private:
    const std::string coord_;
    const double lo_, hi_;
    const int seed_;
    const bool usePatchBounds_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    patchContainer_ = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
  }

  void evaluate();
  
private:
  const std::string coord_;
  const double lo_, hi_;
  const int seed_;
  const bool usePatchBounds_;
  const bool isCoordExpr_; // is this expression evaluating a coordinate or not? for non coordinates, we use the user provided lo and hi
  WasatchCore::UintahPatchContainer* patchContainer_;
  
  ParticleRandomIC( const std::string& coord,
                   const double lo,
                   const double hi,
                   const int seed,
                   const bool usePatchBounds )
  : Expr::Expression<ParticleField>(),
    coord_(coord),
    lo_(lo),
    hi_(hi),
    seed_(seed),
    usePatchBounds_(usePatchBounds),
    isCoordExpr_ ( coord_ != "" )
  {
    this->set_gpu_runnable( true ); 
  }
};

//====================================================================

void
ParticleRandomIC::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  
//  typedef boost::mt19937                       GenT;    // Mersenne Twister
//  typedef boost::normal_distribution<double>   DistT;   // Normal Distribution
//  typedef boost::variate_generator<GenT,DistT> VarGenT;    // Variate generator
//  
//  GenT     eng((unsigned) ( (pid+1) * seed_ * std::time(0) ));
//  DistT    dist(0,1);
//  VarGenT  gen(eng,dist);

  // if this patch has zero memory associated with its particles then just return.
  if (phi.window_with_ghost().local_npts() == 0) return;
  
  double low = lo_;
  double high = hi_;
  if (isCoordExpr_) {
    const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
    const double patchLo  = get_patch_low (patch, coord_);
    const double patchHi  = get_patch_high(patch, coord_);
    low  = usePatchBounds_ ? patchLo : std::max(lo_, patchLo); // disallow creating particles outside the bounds of this patch
    high = usePatchBounds_ ? patchHi : std::min(hi_, patchHi); // disallow creating particles outside the bounds of this patch
  }

  // This is a typedef for a random number generator.
  typedef boost::mt19937 base_generator_type; // mersenne twister
  // Define a random number generator and initialize it with a seed.
  // (The seed is unsigned, otherwise the wrong overload may be selected
  // when using mt19937 as the base_generator_type.)
  // seed the random number generator based on the patch id
  const int pid =  patchContainer_->get_uintah_patch()->getID();
  base_generator_type generator((unsigned) ( (pid + 1) * std::abs(1 + seed_)  )); // 1 + seed to offset a seed = 0
  
  boost::uniform_real<> rand_dist(low,high);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);
  
  ParticleField::iterator phiIter = phi.begin();
  const ParticleField::iterator phiIterEnd = phi.end();
  for( ; phiIter != phiIterEnd; ++phiIter ){
    *phiIter = boost_rand();
  }
}

//==================================================================

/**
 *  \class  ParticleUniformIC
 *  \ingroup WasatchParticles
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Used to uniform initialize partice positions within two bounds. These bounds are either
 specified by the user or inferred from a patch's logical boundaries.
 */
class ParticleUniformIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleUniformIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    /**
     *  \brief Build a ParticleUniformIC expression
     *  \param resultTag  The Tag for the resulting expression that his class computes.
     *  \param lo The lower bound used in the random number generator. This value is
     *  superseded by the lower patch boundaries if usePatchBounds is set to true.
     *  \param hi The upper bound used in the random number generator. This value is
     *  superseded by the upper patch boundary if the usePatchBounds is set to true.
     *  \param transverse A boolean designating whether this coordinate is in the transverse direction.
     *  Given the way particles are laid out in memory, two of the particle coordinates must be
     *  set in the transverse direction so as not to place all the particles on one line. Simply set
     *  this to false on one of the coordinates and true on the other two.
     *  \param coord String denoting the coordinate direction computed by this expression.
     *  Allowed options are "X", "Y", and "Z".
     *  \param usePatchBounds If true, then use the boundaries of the uintah patch on which this
     *  expression is executing.
     */
    Builder( const Expr::Tag& resultTag,
             const double lo,
             const double hi,
             const bool transverse,
             const std::string coord,
             const bool usePatchBounds )
    : ExpressionBuilder(resultTag),
      lo_(lo),
      hi_(hi),
      transverse_(transverse),
      coord_(coord),
      usePatchBounds_(usePatchBounds)
    {}
    
    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleUniformIC( lo_, hi_, transverse_, coord_, usePatchBounds_ );
    }

  private:
    const double lo_, hi_;
    const bool transverse_;
    const std::string coord_;
    const bool usePatchBounds_;
  };
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    patchContainer_ = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
  }

  void evaluate();
  
private:
  const double lo_, hi_;
  const bool transverse_;
  const std::string coord_;
  const bool usePatchBounds_;
  WasatchCore::UintahPatchContainer* patchContainer_;
  
  ParticleUniformIC( const double lo,
                     const double hi,
                     const bool transverse,
                     const std::string coord,
                     const bool usePatchBounds )
  : Expr::Expression<ParticleField>(),
    lo_(lo),
    hi_(hi),
    transverse_(transverse),
    coord_(coord),
    usePatchBounds_(usePatchBounds)
  {
    this->set_gpu_runnable( true );
  }

};

//====================================================================

void
ParticleUniformIC::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  
  double low = lo_, high = hi_;

  if( usePatchBounds_ ){
    const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
    low  = get_patch_low(patch, coord_);
    high = get_patch_high(patch, coord_);
  }

  const double nParticles = phi.window_with_ghost().local_npts();
  const int npart = (int) sqrt(nParticles);
  const double dx = (high-low) / npart;
  ParticleField::iterator phiIter = phi.begin();
  const ParticleField::iterator phiIterEnd = phi.end();
  if( transverse_ ){
  int i = 0;
  int j = 0;
    for( ; phiIter != phiIterEnd; ++phiIter, ++j ){
      const double x = low + j*dx;
      if (x >= high) {
        j = 0;
        ++i;
      }
      *phiIter = low + i*dx;
    }
  }
  else{
    for( int i=0; phiIter != phiIterEnd; ++phiIter, ++i ){
      const double x = low + i*dx;
      if (x >= high) i = 0;
      *phiIter = low + i*dx;
    }
  }
}

//--------------------------------------------------------------------

#endif // ParticlePositionEquation_h
