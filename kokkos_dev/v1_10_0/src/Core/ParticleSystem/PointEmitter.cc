//===========================================================================
//
// Filename: PointEmitter.cc
//
// 
//
//
//
//
//
// Description: PointEmitter implementation
//
//
//===========================================================================

#ifndef __POINTEMITTER_H__
#include <PointEmitter.h>
#endif

PointEmitter::PointEmitter( const Vector& pos )
{
  SetEmitterPosition( pos );
}

PointEmitter::~PointEmitter( void )
{
  // EMPTY
}

Particle*
PointEmitter::EmitParticle( void ) const
{
  Particle* par = new Particle();

  par->SetPosition( GetEmitterPosition() );
  par->SetVelocity( GetNextSpeed() * Rand::vrand() );
  par->SetAge( 0.0 );
  par->SetFixed( false );
  
  return par;
}
