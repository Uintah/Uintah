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



#ifndef GLMATH_H_
#define GLMATH_H_

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
