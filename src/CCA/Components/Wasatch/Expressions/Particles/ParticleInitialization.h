#ifndef ParticleInitialization_h
#define ParticleInitialization_h

#include <expression/ExprLib.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <Core/Grid/Box.h>
//==================================================================
/**
 *  \class  ParticleRandomIC
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Generates a pseudo-random field to initialize particles.
 */
class ParticleRandomIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleRandomIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
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
{}

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
  if (coord == "X") {
    return patch->getBox().lower().x() + patch->dCell().x()/2.0;
  } else if (coord == "Y") {
    return patch->getBox().lower().y() + patch->dCell().y()/2.0;
  } else {
    return patch->getBox().lower().z() + patch->dCell().z()/2.0;
  }
  return 0.0;
}

//--------------------------------------------------------------------

double
get_patch_high(const Uintah::Patch* const patch, const std::string& coord)
{
  if (coord == "X") {
    return patch->getBox().upper().x() - patch->dCell().x()/2.0;
  } else if (coord == "Y") {
    return patch->getBox().upper().y() - patch->dCell().y()/2.0;
  } else {
    return patch->getBox().upper().z() - patch->dCell().z()/2.0;
  }
  return 0.0;
}

//--------------------------------------------------------------------

void
ParticleRandomIC::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  ParticleField::iterator phiIter = phi.begin();
  
  
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
  const int pid =  Uintah::Parallel::getMPIRank();
  base_generator_type generator((unsigned) ( (pid+1) * seed_ * std::time(0) ));
  
  boost::uniform_real<> rand_dist(low,high);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);
  
  while ( phiIter != phi.end() ) {
    *phiIter = boost_rand();
    ++phiIter;
  }
}

//--------------------------------------------------------------------


ParticleRandomIC::Builder::
Builder( const Expr::Tag& result,
        const std::string& coord,
        const double lo,
        const double hi,
        const double seed,
        const bool usePatchBounds )
: ExpressionBuilder(result),
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

//--------------------------------------------------------------------

//==================================================================
/**
 *  \class  ParticleUniformIC
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Generates a pseudo-random field to initialize particles.
 */

class ParticleUniformIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleUniformIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const int nParticles,
            const double lo,
            const double hi,
            const bool transverse,
            const std::string coord,
            const bool usePatchBounds);
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const int nParticles_;
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
  const int nParticles_;
  const double lo_, hi_;
  const bool transverse_;
  const std::string coord_;
  const bool usePatchBounds_;
  Wasatch::UintahPatchContainer* patchContainer_;
  
  ParticleUniformIC( const int nParticles,
                     const double lo,
                     const double hi,
                     const bool transverse,
                     const std::string coord,
                     const bool usePatchBounds);
  
};

//====================================================================

ParticleUniformIC::
ParticleUniformIC(const int nParticles,
                  const double lo,
                  const double hi,
                  const bool transverse,
                  const std::string coord,
                  const bool usePatchBounds)
: Expr::Expression<ParticleField>(),
nParticles_(nParticles),
lo_(lo),
hi_(hi),
transverse_(transverse),
coord_(coord),
usePatchBounds_(usePatchBounds)
{}

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
  ParticleField::iterator phiIter = phi.begin();
  
  double low = lo_, high = hi_;

  if (usePatchBounds_) {
    const Uintah::Patch* const patch = patchContainer_->get_uintah_patch();
    low  = get_patch_low(patch, coord_);
    high = get_patch_high(patch, coord_);
  }
  
  const int npart = (int) sqrt(nParticles_);
  const double dx = (high-low) / npart;
  int i = 0;
  int j = 0;
  if (transverse_) {
    while ( phiIter != phi.end() ) {
      const double x = low + j*dx;
      if (x >= high) {
        j = 0;
        i++;
      }
      *phiIter = low + i*dx;
      ++phiIter;
      j++;
    }
  } else {
    while ( phiIter != phi.end() ) {
      const double x = low + i*dx;
      if (x >= high) i = 0;
      *phiIter = low + i*dx;
      ++phiIter;
      i++;
    }
  }
}

//--------------------------------------------------------------------


ParticleUniformIC::Builder::
Builder( const Expr::Tag& result,
        const int nParticles,
        const double lo,
        const double hi,
        const bool transverse,
        const std::string coord,
        const bool usePatchBounds)
: ExpressionBuilder(result),
nParticles_(nParticles),
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
  return new ParticleUniformIC( nParticles_, lo_, hi_, transverse_, coord_, usePatchBounds_ );
}

//--------------------------------------------------------------------


#endif // ParticlePositionEquation_h
