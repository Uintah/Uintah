//==================================================================
//
// Filename: ParticleSystem.cc
//
//
//
//
//
//
//
//
//
//
//==================================================================

#ifndef __PARTICLESYSTEM_H__
#include <ParticleSystem.h>
#endif

ParticleSystem::ParticleSystem( void ) :
  _numParticles( 0 ),
  _time( 0.0 )
{
  // EMPTY
}

ParticleSystem::~ParticleSystem( void )
{
  for( int i = 0; i < _particles.size(); i++ )
    delete _particles[i];
  _particles.clear();
}

bool
ParticleSystem::Update( float timeStep )
{
  float dt = _dt;
  if( timeStep >= 0.0 )
    dt = timeStep;

  //
  // Update total simulation time
  //
  _time += dt;

  //
  // update all emitter times
  //
  for( int i = 0; i < _emitters.size(); i++ )
    {
      Emitter* emitter = _emitters[i];
      emitter->IncrementTime( dt );
    }
  //
  // update all force times
  //
  for( int i = 0; i < _forces.size(); i++ )
    {
      Force* force = _forces[i];
      force->IncrementTime( dt );
    }

  //
  // For every particle, compute all the forces
  //
  for( int i = 0; i < _particles.size(); i++ ) {
    Particle* par = _particles[i];
    par->ClearForces();
    for( int j = 0; j < _forces.size(); j++ ) {
      Vector force = _forces[i]->GetForce( par->GetPosition() );
      par->AddForce( force );
    }
  }

  //
  // Euler Integration
  // If we want something more complicated like Runge-Kutta, it'll go here
  //
  //
  for( int i = 0; i < _particles.size(); i++ ) {

    //
    // Get derivatives: velocity and acceleration
    //
    Vector vel = _particles[i]->GetVelocity();
    float m = _particles[i]->GetMass();
    Vector force = _particles[i]->GetForce();
    Vector acceleration = ( 1.0 / m ) * force;

    //
    // Scale it by the time step
    //
    acceleration = _dt * acceleration;
    vel = _dt * vel;

    //
    // Collision detection would go here
    //
    //
    
    //
    // Update particles
    //
    _particles[i]->SetPosition( vel + _particles[i]->GetPosition() );
    _particles[i]->SetVelocity( acceleration + _particles[i]->GetVelocity() );
  }
  
  return true;
  
}

bool
ParticleSystem::CreateParticles( Emitter* emitter, int numParticles )
{
  for( int i = 0; i < numParticles; i++ ) {
    Particle* part = emitter->EmitParticle();
    if( part ) {
      _numParticles++;
      _particles.push_back( part );
    }
  }
  return true;
}
      
