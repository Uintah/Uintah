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
