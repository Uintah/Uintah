//===========================================================================
//
// Filename: PointEmitter.h
//
// 
//
//
//
//
//
// Description: PointEmitter interface
//
// Emit particles uniformly in all directions
//
//===========================================================================

#ifndef __POINTEMITTER_H__
#define __POINTEMITTER_H__

#if defined __cplusplus

#ifndef __EMITTER_H__
#include <Emitter.h>
#endif

class PointEmitter : public Emitter
{

public:

  PointEmitter( const Vector& pos );
  //
  // Create a point emitter at a particular position
  //
  //===================================================================
  
  virtual ~PointEmitter( void );

  virtual Particle* EmitParticle( void ) const;
  //
  // Emit a new particle
  //
  //
  //================================================================

protected:

};

#endif
#endif
