//===========================================================================
//
// Filename: Forces.h
//
// 
//
//
//
//
//
// Description: A collection of different forces
//
// Includes: Directional (ie. Gravity or Wind)
//           Turbulence
//           Vortex
//           Torus
//           Drag
//
//===========================================================================

#ifndef __FORCES_H__
#define __FORCES_H__

#if defined __cplusplus

#ifndef __VECTOR_H__
#include <Vector.h>
#endif

#ifndef __PARTICLE_H__
#include <Particle.h>
#endif

#ifndef __FORCE_H__
#include <Force.h>
#endif

//
// This is a directional force such as gravity or unifor wind field
// The magnitude and direction of the force is constant
//

class DirectionalForce : public Force
{

public:
  
  DirectionalForce( const Vector& dir ) : _dir( dir ) { }
  //
  // Construct a directional force in a particular direction
  //
  //
  //=========================================================
  
  virtual ~DirectionalForce( void ) { }

  void SetForce( const Vector& vec ) { _dir = vec; }
  //
  // Set force direction and magnitude
  //
  // For example: gravity would be Vector( 0.0, -9.8, 0.0 )
  //
  //==========================================================
  
  virtual Vector GetForce( const Vector& pos ) const { return _dir; }
  //
  // This is not really correct, because mass is not taken into account
  //
  //==========================================================
  
  virtual Vector GetForce( const Particle* par) const {
    return( par->GetMass() * _dir );
  }
  //
  // F = m * g (or some other acceleration)
  //
  //==========================================================
  
private:

  Vector _dir;

  
};

//
// Drag Force
//
//
//
//

class DragForce : public Force
{

public:
  
  DragForce( const float dragCoeff  = -0.1 ) : _drag( dragCoeff ) { }
  //
  // Create a drag force (resistance to motion) with a specified drag coefficient
  //
  //=============================================================================
  
  virtual ~DragForce( void );

  void SetDirection( const Vector& dir ) {
    _hasVel = true;
    _dir = dir;
    _dir.Normalize();
  }
    
  float GetDragCoefficient( void ) const { return _drag; }
  void SetDragCoefficient( float coeff ) { _drag = coeff; }

  virtual Vector GetForce( const Vector& pos ) const;
  virtual Vector GetForce( const Particle* par ) const;

private:

  bool _hasVel;   // Is drag force set for a particular direction
  float _drag;    // Drag coefficient
  Vector _dir;
  
};

//
//
//
//
//
//

class VortexForce : public Force
{

public:
  
  VortexForce( const Vector& pos, const Vector& dir );
  //
  // Create a vortex force at position and with a direction
  //
  //=====================================================================
  
  virtual ~VortexForce( void );

  virtual Vector GetForce( const Vector& pos ) const;

  void SetSpeed( float speed ) { _speed = speed; }
  
  virtual void IncrementTime( float dt = 1.0f ) {
    float fOldTime = _time;
    _time += dt;
    if( _time >= 180.0f ){
      _time = ((int)dt % 180);
    }
    _angle += _speed;
  }
  
private:

  Vector _dir;
  Vector _pos;
  float _speed;
  float _angle;
  float _radTime;
  
};

//
// Torus force
//
// The torusField node implements an attraction-and-repel field.
//
// The field repels all objects between itself and repelDistance attribute
// and attracts objects greater than attractDistance attribute from itself.
// This will eventually result in the objects clustering in a torus shape around
// the field.
//                                    
// Attributes:
//
// minimum distance from field at which repel is applied: MinDistance
// minimum distance from field at which the force attracts: AttractDistance
// maximum distance from field at which the force repels: RepelDistance
// drag exerted on the attractRepel force: Drag
// amplitude/magnitude of the swarm force: SwarmAmplitude    
// frequency of the swarm force: SwarmFrequency
// phase of the swarm force: SwarmPhase
//
//


class TorusForce : public Force
{

public:

  TorusForce( const Vector& center );

  virtual ~TorusForce( void );

  virtual Vector GetForce( const Vector& pos ) const;

  void SetApplyMAxDistance( bool onoff ) { _applyMaxDistance = onoff; }
  void SetCenter( const Vector& cen ) { _center = cen; }
  void SetMinRepelDistance( float dist ) { _minDistance = dist; }
  void SetMaxRepelDistance( float dist ) { _repelDistance = dist; }
  void SetMinAttractDistance( float dist ) { _attractDistance = dist; }
  void SetDrag( float drag ) { _drag = drag; }
  void SetSwarmAmplitude( float a ) { _swarmAmplitude = a; }
  void SetSwarmFrequency( float a ) { _swarmFrequency = a; }
  void SetSwarmPhase( float a ) { _swarmPhase = a; }

private:

  Vector ApplyMaxDistance( const Vector& pos ) const;
  Vector ApplyNoMaxDistance( const Vector& pos ) const;
  
  void NoiseFunction( float *inNoise, float *out );

  Vector _center;
  float _minDistance;
  float _repelDistance;
  float _attractDistance;
  float _drag;
  float _swarmAmplitude;
  float _swarmPhase;
  float _swarmFrequency;
  bool _applyMaxDistance;
  
};

#endif
#endif
