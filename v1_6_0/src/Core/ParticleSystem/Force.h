//===========================================================================
//
// Filename: Force.h
//
// 
//
//
//
//
//
// Description: Force interface
//
//
//===========================================================================

#ifndef __FORCE_H__
#define __FORCE_H__

#if defined __cplusplus

class Vector;
class Particle;

class Force
{

public:
  
  Force( void ) { }

  virtual ~Force( void ) { }

  virtual Vector GetForce( const Vector& pos ) const = 0;
  //
  // Get the force that is acting at particular position pos
  //
  // Every class derived from the Force class must implement this!
  //
  //========================================================================

  virtual Vector GetForce( const Particle* particle ) const = 0;
  //
  // Get the force that is acting on a particle 
  //
  // Every class derived from the Force class must implement this!
  //
  //========================================================================

  void SetTime( float time ) { _time = _time; }
  float GetTime( void ) const { return _time; }

  virtual void IncrementTime( float dTime = 1.0f ) { _time = _time + dTime; }
  //
  // Increment time of the force
  //
  //
  //========================================================================
  
protected:

  float _time;
  
};

#endif
#endif
