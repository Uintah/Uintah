/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//===========================================================================
//
// Filename: Emitter.h
//
// 
//
//
//
//
//
// Description: Emitter interface
//
//
//===========================================================================

#ifndef __EMITTER_H__
#define __EMITTER_H__

#if defined __cplusplus

#ifndef __VECTOR_H__
#include <Vector.h>
#endif

#ifndef __RAND_H__
#include <Rand.h>
#endif

#ifndef __PARTICLE_H__
#include <Particle.h>
#endif


class Emitter
{

public:

  Emitter( void );
  Emitter( const Emitter& part );
  
  virtual ~Emitter( void );

  void SetEmitterPosition( const Vector& vec ) { if( _fixed == false ) { _pos = vec; } }
  void SetEmitterVelocity( const Vector& vec ) { if( _fixed == false ) { _vel = vec; } }
  void SetFixed( bool onoff ) { _fixed = onoff; }
  //
  // Set particle quantities
  //
  //=======================================================

  const Vector& GetEmitterPosition( void ) const { return _pos; }
  const Vector& GetEmitterVelocity( void ) const { return _vel; }
  bool IsFixed( void ) const { return _fixed; }
  //
  // Get particle quantities
  //
  //=======================================================

  virtual Particle* EmitParticle( void ) const = 0;
  //
  // Emit a new particle
  //
  // Memory is allocated, but it's user's responsibility to
  // delete memory afterwards
  //
  //
  //================================================================

  float GetRate( void ) const { return _rate; }
  void SetRate( float rate ) { _rate = rate; }
  //
  // Set the rate at which particles are emitted
  //
  //================================================================

  long GetSeed( void ) const { return _seed; }
  void SetSeed( long seed ) { _seed = seed; }
  //
  // Set the random number generator seed for this emitter
  //
  //================================================================

  void SetSpeed( float speed ) { _speed = speed; }
  float GetSpeed( void ) const { return _speed; }
  //
  // Set the speed with which the particles are emitted
  //
  //================================================================

  void SetSpeedVariance( float speed ) { _speedVar = speed; }
  float GetSpeedVariance( void ) const { return _speedVar; }
  //
  // Set the speed variance with which the particles are emitted
  //
  //================================================================

  virtual void IncrementTime( float dt ) { _time += dt; }
  //
  // Increment time
  //
  //================================================================
  
protected:

  float GetNextSpeed( void ) const {
    return Rand::frand( _speed, _speedVar );
  }

  Vector _pos;   // Position of the emitter
  Vector _vel;   // Velocity of the emitter
  bool _fixed;   // Is this emitter fixed
  float _speed;  // Speed with which the particles are emitted
  float _speedVar; // Speed variance
  long _seed;    // Random number generator seed
  float _rate;   // Emission rate

  float _time;
  
};

#endif
#endif
