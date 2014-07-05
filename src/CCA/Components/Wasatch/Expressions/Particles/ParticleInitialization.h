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

//==================================================================

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
     These can only be X, Y, or Z.
     *
     *  \param lo         The lower bound used in the random number generator. This value is
     superseeded by the lower patch boundaries if usePatchBounds is set to true.
     *
     *  \param hi         The upper bound used in the random number generator. This value is
     superseeded by the upper patch boundary if the usePatchBounds is set to true.
     *
     *  \param seed       The seed for the random number generator. This is a required quantity.
     *
     *  \param usePatchBounds If true, then use the boundaries of the uintah patch on which this
     expression is executing.
     */
    Builder( const Expr::Tag& resultTag,
            const std::string& coord,
            const double lo,
            const double hi,
            const double seed,
            const bool usePatchBounds);
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const std::string coord_;
    const double lo_, hi_, seed_;
    const bool usePatchBounds_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  const std::string coord_;
  const double lo_, hi_, seed_;
  const bool usePatchBounds_;
  Wasatch::UintahPatchContainer* patchContainer_;
  
  ParticleRandomIC( const std::string& coord,
                   const double lo,
                   const double hi,
                   const double seed,
                   const bool usePatchBounds );
};

//====================================================================

ParticleRandomIC::
ParticleRandomIC(const std::string& coord,
                 const double lo,
                 const double hi,
                 const double seed,
                 const bool usePatchBounds )
: Expr::Expression<ParticleField>(),
coord_(coord),
lo_(lo),
hi_(hi),
seed_(seed),
usePatchBounds_(usePatchBounds)
{
  this->set_gpu_runnable( false );  // definitely not GPU ready
}

//--------------------------------------------------------------------

void
ParticleRandomIC::
advertise_dependents( Expr::ExprDeps& exprDeps )
{}

//--------------------------------------------------------------------

void
ParticleRandomIC::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  patchContainer_ = opDB.retrieve_operator<Wasatch::UintahPatchContainer>();
}

//--------------------------------------------------------------------

void
ParticleRandomIC::
bind_fields( const Expr::FieldManagerList& fml )
{}

//--------------------------------------------------------------------

double
get_patch_low(const Uintah::Patch* const patch, const std::string& coord)
{
  if     ( coord == "X" ) return patch->getBox().lower().x() + patch->dCell().x()/2.0;
  else if( coord == "Y" ) return patch->getBox().lower().y() + patch->dCell().y()/2.0;
  else if( coord == "Z" ) return patch->getBox().lower().z() + patch->dCell().z()/2.0;
  assert( false ); // should never get here.
  return 0.0;
}

//--------------------------------------------------------------------

double
get_patch_high(const Uintah::Patch* const patch, const std::string& coord)
{
  if     ( coord == "X" ) return patch->getBox().upper().x() - patch->dCell().x()/2.0;
  else if( coord == "Y" ) return patch->getBox().upper().y() - patch->dCell().y()/2.0;
  else if( coord == "Z" ) return patch->getBox().upper().z() - patch->dCell().z()/2.0;
  assert( false ); // should never get here
  return 0.0;
}

//--------------------------------------------------------------------

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

  double low = lo_, high = hi_;
  if (usePatchBounds_) {
    const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
    low  = get_patch_low(patch, coord_);
    high = get_patch_high(patch, coord_);
  }

  // This is a typedef for a random number generator.
  typedef boost::mt19937 base_generator_type; // mersenne twister
  // Define a random number generator and initialize it with a seed.
  // (The seed is unsigned, otherwise the wrong overload may be selected
  // when using mt19937 as the base_generator_type.)
  // seed the random number generator based on the MPI rank
  const int pid =  patchContainer_->get_uintah_patch()->getID();
  base_generator_type generator((unsigned) ( (pid+1) * seed_ * std::time(0) ));
  
  boost::uniform_real<> rand_dist(low,high);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);
  
  ParticleField::iterator phiIter = phi.begin();
  ParticleField::iterator phiIterEnd = phi.end();
  for( ; phiIter != phiIterEnd; ++phiIter ){
    *phiIter = boost_rand();
  }
}

//--------------------------------------------------------------------

ParticleRandomIC::Builder::
Builder( const Expr::Tag& resultTag,
        const std::string& coord,
        const double lo,
        const double hi,
        const double seed,
        const bool usePatchBounds )
: ExpressionBuilder(resultTag),
coord_(coord),
lo_(lo),
hi_(hi),
seed_(seed),
usePatchBounds_(usePatchBounds)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
ParticleRandomIC::Builder::build() const
{
  return new ParticleRandomIC(coord_,lo_, hi_, seed_,usePatchBounds_ );
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
     *  \brief            Build a ParticleUniformIC expression
     *  \param resultTag  The Tag for the resulting expression that his class computes.
     *  \param lo         The lower bound used in the random number generator. This value is
     superseeded by the lower patch boundaries if usePatchBounds is set to true.
     *  \param hi         The upper bound used in the random number generator. This value is
     superseeded by the upper patch boundary if the usePatchBounds is set to true.
     *  \param transverse A boolean designating whether this coordinate is in the transverse direction.
     Given the way particles are laid out in memory, two of the particle coordinates must be
     set in the transverse direction so as not to place all the particles on one line. Simply set
     this to false on one of the coordinates and true on the other two.
     *  \param seed       The seed for the random number generator. This is a required quantity.
     *  \param usePatchBounds If true, then use the boundaries of the uintah patch on which this
     expression is executing.
     */
    Builder( const Expr::Tag& resultTag,
            const double lo,
            const double hi,
            const bool transverse,
            const std::string coord,
            const bool usePatchBounds);
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double lo_, hi_;
    const bool transverse_;
    const std::string coord_;
    const bool usePatchBounds_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  const double lo_, hi_;
  const bool transverse_;
  const std::string coord_;
  const bool usePatchBounds_;
  Wasatch::UintahPatchContainer* patchContainer_;
  
  ParticleUniformIC( const double lo,
                     const double hi,
                     const bool transverse,
                     const std::string coord,
                     const bool usePatchBounds);
};

//====================================================================

ParticleUniformIC::
ParticleUniformIC(const double lo,
                  const double hi,
                  const bool transverse,
                  const std::string coord,
                  const bool usePatchBounds)
: Expr::Expression<ParticleField>(),
lo_(lo),
hi_(hi),
transverse_(transverse),
coord_(coord),
usePatchBounds_(usePatchBounds)
{
  this->set_gpu_runnable( false );
}

//--------------------------------------------------------------------

void
ParticleUniformIC::
advertise_dependents( Expr::ExprDeps& exprDeps )
{}

//--------------------------------------------------------------------

void
ParticleUniformIC::
bind_fields( const Expr::FieldManagerList& fml )
{}

//--------------------------------------------------------------------

void
ParticleUniformIC::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  patchContainer_ = opDB.retrieve_operator<Wasatch::UintahPatchContainer>();
}

//--------------------------------------------------------------------

void
ParticleUniformIC::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  
  double low = lo_, high = hi_;

  if (usePatchBounds_) {
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

ParticleUniformIC::Builder::
Builder( const Expr::Tag& resultTag,
        const double lo,
        const double hi,
        const bool transverse,
        const std::string coord,
        const bool usePatchBounds)
: ExpressionBuilder(resultTag),
lo_(lo),
hi_(hi),
transverse_(transverse),
coord_(coord),
usePatchBounds_(usePatchBounds)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
ParticleUniformIC::Builder::build() const
{
  return new ParticleUniformIC( lo_, hi_, transverse_, coord_, usePatchBounds_ );
}

//--------------------------------------------------------------------

#endif // ParticlePositionEquation_h
