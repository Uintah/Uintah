//===========================================================================
//
// Filename: DirectionalEmitter.cc
//
// 
//
//
//
//
//
// Description: DirectionalEmitter implementation
//
// Emit particles in a particular direction
//
//===========================================================================

#ifndef __DIRECTIONALEMITTER_H__
#include <DirectionalEmitter.h>
#endif

DirectionalEmitter::DirectionalEmitter( const Vector& pos,
					const Vector& direction ) :
  _dir( direction )
{
  SetEmitterPosition( pos );
}

DirectionalEmitter::~DirectionalEmitter( void )
{
  // EMPTY
}

Particle*
DirectionalEmitter::EmitParticle( void ) const
{
  Particle* par = new Particle();

  par->SetPosition( GetEmitterPosition() );
  par->SetVelocity( GetNextSpeed() * _dir );
  par->SetAge( 0.0 );
  par->SetFixed( false );
  
  return par;
}
