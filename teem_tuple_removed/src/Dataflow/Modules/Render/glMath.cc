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


#include <math.h>
#include "glMath.h"

/* 4x4 unit matrix. */

void glEye( GLfloat m[16] )
{
  for( int i=0; i<16; i++ )
    {
      if( i%5 == 0 )
	m[i] = 1.0;
      else
	m[i] = 0.0;
    }
}


/* 4x4 matrix-vector multiplication. */

void glVectorMult( GLfloat mxv[16], GLfloat ma[16], GLfloat v[4] )
{
  mxv[0] = ma[0]*v[0] + ma[4]*v[1] + ma[8]*v[2] + ma[12]*v[3];
  mxv[1] = ma[1]*v[0] + ma[5]*v[1] + ma[9]*v[2] + ma[13]*v[3];
  mxv[2] = ma[2]*v[0] + ma[6]*v[1] + ma[10]*v[2] + ma[14]*v[3];
  mxv[3] = ma[3]*v[0] + ma[7]*v[1] + ma[11]*v[2] + ma[15]*v[3];
}


/* 4x4 matrix-matrix multiplication. */

void glMatrixMult( GLfloat maxb[16], GLfloat ma[16], GLfloat mb[16] )
{
  maxb[0] = ma[0]*mb[0] + ma[4]*mb[1] + ma[8]*mb[2] + ma[12]*mb[3];
  maxb[1] = ma[1]*mb[0] + ma[5]*mb[1] + ma[9]*mb[2] + ma[13]*mb[3];
  maxb[2] = ma[2]*mb[0] + ma[6]*mb[1] + ma[10]*mb[2] + ma[14]*mb[3];
  maxb[3] = ma[3]*mb[0] + ma[7]*mb[1] + ma[11]*mb[2] + ma[15]*mb[3];
  
  maxb[4] = ma[0]*mb[4] + ma[4]*mb[5] + ma[8]*mb[6] + ma[12]*mb[7];
  maxb[5] = ma[1]*mb[4] + ma[5]*mb[5] + ma[9]*mb[6] + ma[13]*mb[7];
  maxb[6] = ma[2]*mb[4] + ma[6]*mb[5] + ma[10]*mb[6] + ma[14]*mb[7];
  maxb[7] = ma[3]*mb[4] + ma[7]*mb[5] + ma[11]*mb[6] + ma[15]*mb[7];
  
  maxb[8] = ma[0]*mb[8] + ma[4]*mb[9] + ma[8]*mb[10] + ma[12]*mb[11];
  maxb[9] = ma[1]*mb[8] + ma[5]*mb[9] + ma[9]*mb[10] + ma[13]*mb[11];
  maxb[10] = ma[2]*mb[8] + ma[6]*mb[9] + ma[10]*mb[10] + ma[14]*mb[11];
  maxb[11] = ma[3]*mb[8] + ma[7]*mb[9] + ma[11]*mb[10] + ma[15]*mb[11];
  
  maxb[12] = ma[0]*mb[12] + ma[4]*mb[13] + ma[8]*mb[14] + ma[12]*mb[15];
  maxb[13] = ma[1]*mb[12] + ma[5]*mb[13] + ma[9]*mb[14] + ma[13]*mb[15];
  maxb[14] = ma[2]*mb[12] + ma[6]*mb[13] + ma[10]*mb[14] + ma[14]*mb[15];
  maxb[15] = ma[3]*mb[12] + ma[7]*mb[13] + ma[11]*mb[14] + ma[15]*mb[15];
}


/* Homogeneous rigid direct transform. */

void glTransform( GLfloat p[3], GLfloat m[16], GLfloat p0[3] )
{
  p[0] = m[0]*p0[0] + m[4]*p0[1] + m[8]*p0[2] + m[12];
  p[1] = m[1]*p0[0] + m[5]*p0[1] + m[9]*p0[2] + m[13];
  p[2] = m[2]*p0[0] + m[6]*p0[1] + m[10]*p0[2] + m[14];
}


/* Homogeneous rigid inverse transform. */

void glInvTransform( GLfloat p[3], GLfloat m[16], GLfloat p0[3] )
{
  GLfloat tmp[3];

  tmp[0] = p0[0] - m[12];
  tmp[1] = p0[1] - m[13];
  tmp[2] = p0[2] - m[14];

  p[0] = m[0]*tmp[0] + m[1]*tmp[1] + m[2]*tmp[2];
  p[1] = m[4]*tmp[0] + m[5]*tmp[1] + m[6]*tmp[2];
  p[2] = m[8]*tmp[0] + m[9]*tmp[1] + m[10]*tmp[2];
}


/* Simplified 4x4 homogeneous inverse. */

void glInverse( GLfloat invm[16], GLfloat m[16] )
{
  GLfloat det = 
    m[0]*m[5]*m[10]-
    m[0]*m[6]*m[9]-
    m[1]*m[4]*m[10]+
    m[1]*m[6]*m[8]+
    m[2]*m[4]*m[9]-
    m[2]*m[5]*m[8];

  invm[0] = (m[5]*m[10]-m[6]*m[9])/det;
  invm[1] = (-m[1]*m[10]+m[2]*m[9])/det;
  invm[2] = (m[1]*m[6]-m[2]*m[5])/det;
  invm[3] = 0.0;

  invm[4] = (-m[4]*m[10]+m[6]*m[8])/det;
  invm[5] = (m[0]*m[10]-m[2]*m[8])/det;
  invm[6] = (-m[0]*m[6]+m[2]*m[4])/det;
  invm[7] = 0.0;

  invm[8] = (m[4]*m[9]-m[5]*m[8])/det;
  invm[9] = (-m[0]*m[9]+m[1]*m[8])/det;
  invm[10] = (m[0]*m[5]-m[1]*m[4])/det;
  invm[11] = 0.0;
  
  invm[12] = (-m[4]*m[9]*m[14]+m[4]*m[13]*m[10]+
	      m[5]*m[8]*m[14]-m[5]*m[12]*m[10]-
	      m[6]*m[8]*m[13]+m[6]*m[12]*m[9])/det;
  invm[13] = (m[0]*m[9]*m[14]-m[0]*m[13]*m[10]-
	      m[1]*m[8]*m[14]+m[1]*m[12]*m[10]+
	      m[2]*m[8]*m[13]-m[2]*m[12]*m[9])/det;
  invm[14] = (-m[0]*m[5]*m[14]+m[0]*m[13]*m[6]+
	      m[1]*m[4]*m[14]-m[1]*m[12]*m[6]-
	      m[2]*m[4]*m[13]+m[2]*m[12]*m[5])/det;
  invm[15] = 1.0;
}


void zyx2R( GLfloat m[16], GLfloat phi, GLfloat theta, GLfloat psi )
{
  GLfloat ca = cos(phi), sa = sin(phi);
  GLfloat cb = cos(theta),  sb = sin(theta);
  GLfloat cy = cos(psi), sy = sin(psi);

  m[0] = ca*cb;
  m[4] = ca*sb*sy - sa*cy;
  m[8] = ca*sb*cy + sa*sy;
  m[12] = 0.0;

  m[1] = sa*cb;
  m[5] = sa*sb*sy + ca*cy;
  m[9] = sa*sb*cy - ca*sy;
  m[13] = 0.0;

  m[2] = -sb;
  m[6] = cb*sy;
  m[10] = cb*cy;
  m[14] = 0.0;

  m[3] = 0.0;
  m[7] = 0.0;
  m[11] = 0.0;
  m[15] = 1.0;
}

/*
 * Convert rotation matrix to zyx euler angles.
 */

void R2zyx( GLfloat* phi, GLfloat* theta, GLfloat* psi, GLfloat m[16] )
{
  GLfloat cphi, sphi, cpsi, spsi;

  if( fabs(m[0]) > 0.0 )
    {
      *phi = atan(m[1]/m[0]);
      cphi = cos(*phi); sphi = sin(*phi);

      *psi = atan2(-m[9]*cphi+m[8]*sphi, m[5]*cphi-m[4]*sphi);
      cpsi = cos(*psi); spsi = sin(*psi);

      *theta = atan2(-m[2], m[10]*cpsi+m[6]*spsi);
    }
  else
    {
      if( fabs(m[1]) > 0.0 )
        {
          *phi = M_PI_2;
          *psi = atan2(m[8], -m[4]);
          cpsi = cos(*psi); spsi = sin(*psi);
          *theta = atan2(-m[2], m[10]*cpsi+m[6]*spsi);
        }
      else
        {

        /* Resolve degenerate case */

          if( m[2] > 0.0 )
            {
              *theta = -M_PI_2;
              *phi = atan2(-m[4], m[5]);
              *psi = 0.0;
            }
          else
            {
              *theta = M_PI_2;
              *phi = -atan2(m[4], m[5]);
              *psi = 0.0;
            }
        }
    }
}


/* Angle-axis formula. */

void glAxis2Rot( GLfloat m[16], GLfloat k[3], GLfloat theta )
{
  float c = cos(theta);
  float s = sin(theta);
  float v = 1 - c;

  m[0] = k[0]*k[0]*v + c;
  m[4] = k[0]*k[1]*v - k[2]*s;
  m[8] = k[0]*k[2]*v + k[1]*s;

  m[1] = k[0]*k[1]*v + k[2]*s;
  m[5] = k[1]*k[1]*v + c;
  m[9] = k[1]*k[2]*v - k[0]*s;

  m[2] = k[0]*k[2]*v - k[1]*s;
  m[6] = k[1]*k[2]*v + k[0]*s;
  m[10] = k[2]*k[2]*v + c;
}


/* Inverse angle-axis formula. */

void glRot2Axis( GLfloat k[3], GLfloat *theta, GLfloat m[16] )
{
  GLfloat c = 0.5 * (m[0] + m[5] + m[10] - 1.0);
  GLfloat r1 = m[6] - m[9];
  GLfloat r2 = m[8] - m[2];
  GLfloat r3 = m[1] - m[4];
  GLfloat s = 0.5 * sqrt(r1*r1+r2*r2+r3*r3);

  *theta = atan2(s, c);

  if( fabs(s) > GLEPSILON )
    {
      c = 2.0*s;

      k[0] = r1 / c;
      k[1] = r2 / c;
      k[2] = r3 / c;
    }
  else
    {
      k[0] = 0;
      k[1] = 0;
      k[2] = 1;
    }
}

