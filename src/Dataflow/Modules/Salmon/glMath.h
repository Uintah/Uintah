
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
