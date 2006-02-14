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
