//===========================================================================
//
// Filename: Particle.h
//
// 
//
//
//
//
//
// Description: Particle interface
//
//
//===========================================================================

#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#if defined __cplusplus

#ifndef __VECTOR_H__
#include <Vector.h>
#endif

class Particle
{

public:

  Particle( void );
  Particle( const Particle& part );
  
  virtual ~Particle( void );

  void SetMass( float mass ) { _m = mass; }
  void SetPosition( const Vector& vec ) { if( _fixed == false ) { _pos = vec; } }
  void SetVelocity( const Vector& vec ) { if( _fixed == false ) { _vel = vec; } }
  void SetForce( const Vector& vec ) { _force = vec; }
  void SetFixed( bool onoff ) { _fixed = onoff; }
  void SetAge( long age ) { _age = age; }
  //
  // Set particle quantities
  //
  //=======================================================

  float GetMass( void ) const { return _m; }
  const Vector& GetPosition( void ) const { return _pos; }
  const Vector& GetVelocity( void ) const { return _vel; }
  const Vector& GetForce( void ) const { return _force; }
  bool IsFixed( void ) const { return _fixed; }
  long GetAge( void ) const { return _age; }
  //
  // Get particle quantities
  //
  //=======================================================

  void AddForce( const Vector& vec ) { _force = _force + vec; }
  void ClearForces( void );
  //
  // Clear all forces acting on the particle to 0
  //
  //=======================================================

  void IncrementAge( long incr ) { _age += incr; }
  //
  // Increment the age of a particle
  //
  //=======================================================
  
private:

  float _m;      // Mass
  Vector _pos;   // Position
  Vector _vel;   // Velocity
  Vector _force; // Force 
  bool _fixed;   // Is this particle in a fixed position?
  long _age;     // Age of the particle
  
};

#endif
#endif
