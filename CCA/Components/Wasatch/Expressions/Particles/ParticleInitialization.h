#ifndef ParticleInitialization_h
#define ParticleInitialization_h

#include <expression/ExprLib.h>

#include <spatialops/OperatorDatabase.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/particles/ParticleFieldTypes.h>

//==================================================================
/**
 *  \class  ParticleRandomIC
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Generates a pseudo-random field to initialize particles.
 */
template< typename GridCoordT >
class ParticleRandomIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleRandomIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const double lo,
            const double hi,
            const Expr::Tag& exprLoHiTag,
            const double seed );
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const double lo_, hi_, seed_;
    const Expr::Tag exprLoHiTag_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  const double lo_, hi_, seed_;
  const Expr::Tag exprLoHiTag_;
  const GridCoordT *x_;
  
  ParticleRandomIC( const double lo,
              const double hi,
              const Expr::Tag& exprLoHiTag,
              const double seed );
  
};

//====================================================================

template<typename GridCoordT>
ParticleRandomIC<GridCoordT>::
ParticleRandomIC(const double lo,
            const double hi,
            const Expr::Tag& exprLoHiTag,
            const double seed )
: Expr::Expression<ParticleField>(),
lo_(lo),
hi_(hi),
exprLoHiTag_(exprLoHiTag),
seed_(seed)
{}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleRandomIC<GridCoordT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if (exprLoHiTag_ != Expr::Tag()) {
    exprDeps.requires_expression( exprLoHiTag_ );
  }
}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleRandomIC<GridCoordT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<GridCoordT>::type& fm = fml.template field_manager<GridCoordT>();
  if (exprLoHiTag_ != Expr::Tag()) {
    x_ = &fm.field_ref( exprLoHiTag_ );
  } else {
    x_ = NULL;
  }
}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleRandomIC<GridCoordT>::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  typename ParticleField::iterator phiIter = phi.begin();
  
  
//  typedef boost::mt19937                       GenT;    // Mersenne Twister
//  typedef boost::normal_distribution<double>   DistT;   // Normal Distribution
//  typedef boost::variate_generator<GenT,DistT> VarGenT;    // Variate generator
//  
//  GenT     eng((unsigned) ( (pid+1) * seed_ * std::time(0) ));
//  DistT    dist(0,1);
//  VarGenT  gen(eng,dist);

  
  // This is a typedef for a random number generator.
  typedef boost::mt19937 base_generator_type; // mersenne twister
  // Define a random number generator and initialize it with a seed.
  // (The seed is unsigned, otherwise the wrong overload may be selected
  // when using mt19937 as the base_generator_type.)
  // seed the random number generator based on the MPI rank
  const int pid =  Uintah::Parallel::getMPIRank();
  base_generator_type generator((unsigned) ( (pid+1) * seed_ * std::time(0) ));
  const double low  = x_ ? field_min(*x_) : lo_;
  const double high = x_ ? field_max(*x_) : hi_;
  
  boost::uniform_real<> rand_dist(low,high);
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > boost_rand(generator, rand_dist);
  
  while ( phiIter != phi.end() ) {
    *phiIter = boost_rand();
    ++phiIter;
  }
}

//--------------------------------------------------------------------

template< typename GridCoordT >
ParticleRandomIC<GridCoordT>::Builder::
Builder( const Expr::Tag& result,
        const double lo,
        const double hi,
        const Expr::Tag& exprLoHiTag,
        const double seed )
: ExpressionBuilder(result),
lo_(lo),
hi_(hi),
exprLoHiTag_(exprLoHiTag),
seed_(seed)
{}

//--------------------------------------------------------------------

template< typename GridCoordT >
Expr::ExpressionBase*
ParticleRandomIC<GridCoordT>::Builder::build() const
{
  return new ParticleRandomIC<GridCoordT>(lo_, hi_, exprLoHiTag_, seed_ );
}

//--------------------------------------------------------------------

//==================================================================
/**
 *  \class  ParticleUniformIC
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Generates a pseudo-random field to initialize particles.
 */
template< typename GridCoordT >
class ParticleUniformIC : public Expr::Expression<ParticleField>
{
public:
  
  /**
   *  \brief Builds a ParticleUniformIC expression.
   */
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder( const Expr::Tag& result,
            const Expr::Tag& exprLoHiTag,
            const int nParticles,
            const bool transverse);
    
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag exprLoHiTag_;
    const int nParticles_;
    const bool transverse_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  const Expr::Tag exprLoHiTag_;
  const int nParticles_;
  const bool transverse_;
  const GridCoordT *x_;
  
  ParticleUniformIC( const Expr::Tag& exprLoHiTag,
                     const int nParticles,
                     const bool transverse);
  
};

//====================================================================

template<typename GridCoordT>
ParticleUniformIC<GridCoordT>::
ParticleUniformIC(const Expr::Tag& exprLoHiTag,
                  const int nParticles,
                  const bool transverse)
: Expr::Expression<ParticleField>(),
exprLoHiTag_(exprLoHiTag),
nParticles_(nParticles),
transverse_(transverse)
{}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleUniformIC<GridCoordT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( exprLoHiTag_ );
}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleUniformIC<GridCoordT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<GridCoordT>::type& fm = fml.template field_manager<GridCoordT>();
  x_ = &fm.field_ref( exprLoHiTag_ );
}

//--------------------------------------------------------------------

template< typename GridCoordT >
void
ParticleUniformIC<GridCoordT>::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& phi = this->value();
  typename ParticleField::iterator phiIter = phi.begin();
  
  const double low  = field_min_interior(*x_);
  const double high = field_max_interior(*x_);
  
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

template< typename GridCoordT >
ParticleUniformIC<GridCoordT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& exprLoHiTag,
        const int nParticles,
        const bool transverse)
: ExpressionBuilder(result),
exprLoHiTag_(exprLoHiTag),
nParticles_(nParticles),
transverse_(transverse)
{}

//--------------------------------------------------------------------

template< typename GridCoordT >
Expr::ExpressionBase*
ParticleUniformIC<GridCoordT>::Builder::build() const
{
  return new ParticleUniformIC<GridCoordT>(exprLoHiTag_, nParticles_, transverse_ );
}

//--------------------------------------------------------------------


#endif // ParticlePositionEquation_h
