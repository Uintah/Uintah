//===========================================================================
//
// Filename: Particle.cc
//
// 
//
//
//
//
//
// Description: Particle implementation
//
//
//===========================================================================

#ifndef __PARTICLE_H__
#include <Particle.h>
#endif

Particle::Particle( void ) :
  _m( 0.0 ),
  _pos( Vector( 0.0, 0.0, 0.0 ) ),
  _vel( Vector( 0.0, 0.0, 0.0 ) ),
  _force( Vector( 0.0, 0.0, 0.0 ) ),
  _fixed( false ),
  _age( 0 )
{
  // EMPTY
}

Particle::Particle( const Particle& part )
{
  _m = part._m;
  _pos = part._pos;
  _vel = part._vel;
  _force = part._force;
  _fixed = part._fixed;
  _age = part._age;

}
  
Particle::~Particle( void )
{
  // EMPTY
}

void
Particle::ClearForces( void )
{
  _force = Vector( 0.0, 0.0, 0.0 );
}
