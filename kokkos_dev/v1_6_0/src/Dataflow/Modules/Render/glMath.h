/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


#ifndef GLMATH_H_
#define GLMATH_H_

#include <GL/gl.h>
#define GLEPSILON 1e-7

void glEye( GLfloat m[16] );
void glVectorMult( GLfloat mxv[3], GLfloat m[16], GLfloat v[4] );
void glMatrixMult( GLfloat maxb[16], GLfloat ma[16], GLfloat mb[16] );
void glTransform( GLfloat p[3], GLfloat m[16], GLfloat p0[3] );
void glInvTransform( GLfloat p[3], GLfloat m[16], GLfloat p0[3] );
void glInverse( GLfloat invm[16], GLfloat m[16] );
void zyx2R( GLfloat m[16], GLfloat phi, GLfloat theta, GLfloat psi );
void R2zyx( GLfloat* phi, GLfloat* theta, GLfloat* psi, GLfloat m[16] );
void glAxis2Rot( GLfloat m[16], GLfloat k[3], GLfloat theta );
void glRot2Axis( GLfloat k[3], GLfloat *theta, GLfloat m[16] );

#endif /* GL_MATH_H_ */
