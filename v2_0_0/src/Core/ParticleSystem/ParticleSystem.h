//==================================================================
//
// Filename: ParticleSystem.h
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
#define __PARTICLESYSTEM_H__

#if defined __cplusplus

#ifndef __EMITTER_H__
#include <Emitter.h>
#endif

#ifndef __FORCE_H__
#include <Force.h>
#endif

#ifndef __PARTICLE_H__
#include <Particle.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

using namespace std;

class ParticleSystem
{

public:

  ParticleSystem( void );

  virtual ~ParticleSystem( void );
  
  int GetNumParticles( void ) const { return _numParticles; }

  float GetTime( void ) const { return _time; }
  void SetTimeStep( float timeStep ) { _dt = timeStep; }
  
  void AddForce( Force* force ) { _forces.push_back( force ); }
  void AddEmitter( Emitter* emitter ) { _emitters.push_back( emitter ); }

  virtual bool CreateParticles( Emitter* emitter, int numParticles );
  //
  // Create some number of particles
  //
  // Number of particles is created with a given emitter
  // This implies that particles are generated a priori and then number of
  // particles during the simulation doesn't change
  // 
  //============================================================================
  
  virtual bool Update( float timeStep = -1.0 );
  //
  // Compute particle system in the next time step
  // If not time step is specified, the default (global) time step is used
  //
  // Particles positions and velocities are updated as well as
  // forces and emitters (if they are moving)
  //
  // Return 'true' if successful
  //
  //=============================================================================
  
private:

  float _time;
  float _dt;
  int _numParticles;
  vector<Force*> _forces;
  vector<Emitter*> _emitters;
  vector<Particle*> _particles;
  
};

#endif
#endif
