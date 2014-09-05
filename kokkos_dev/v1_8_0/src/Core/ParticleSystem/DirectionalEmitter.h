//===========================================================================
//
// Filename: DirectionalEmitter.h
//
// 
//
//
//
//
//
// Description: DirectionalEmitter interface
//
// Emit particles in a particular direction
//
//===========================================================================

#ifndef __DIRECTIONALEMITTER_H__
#define __DIRECTIONALEMITTER_H__

#if defined __cplusplus

#ifndef __EMITTER_H__
#include <Emitter.h>
#endif

class DirectionalEmitter : public Emitter
{

public:

  DirectionalEmitter( const Vector& pos, const Vector& direction );
  //
  // Create a directional emitter at a particular position
  // that emits partical in a specified direction
  //
  //===================================================================
  
  virtual ~DirectionalEmitter( void );

  void SetDirection( const Vector& dir ) { _dir = dir; }
  const Vector& GetDirection( void ) const { return _dir; }
  
  virtual Particle* EmitParticle( void ) const;
  //
  // Emit a new particle
  //
  //
  //================================================================

private:

  Vector _dir;
  
};

#endif
#endif
