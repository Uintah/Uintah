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

/*
  Thevx contents of this file are subject to the University of Utah Public
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

#include <sci_defs/chromium_defs.h>

#include <Core/GLVolumeRenderer/GLVolRenState.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/Geom/GeomOpenGL.h>

#include <vector>
#include <stdlib.h>
#include <iostream>
using namespace std;


static const char* ShaderString =
"!!ARBfp1.0 \n\
TEMP c0, c; \n\
ATTRIB fc = fragment.color; \n\
ATTRIB tf = fragment.texcoord[0]; \n\
TEX c0, tf, texture[0], 3D; \n\
TEX c, c0, texture[1], 1D; \n\
MUL c, c, fc; \n\
MOV_SAT result.color, c; \n\
END";

/*
// static const char* ShaderString =
// "!!ARBfp1.0 \n\
// TEMP c0, c; \n\
// PARAM p = state.fog.params; \n\
// PARAM fogColor = state.fog.color; \n\
// TEMP fogFactor; \n\
// ATTRIB fogCoord = fragment.fogcoord; \n\
// MAD_SAT fogFactor.x, p.y, fogCoord.x, p.z; \n\
// ATTRIB fc = fragment.color; \n\
// ATTRIB tf = fragment.texcoord[0]; \n\
// TEX c0, tf, texture[0], 3D; \n\
// TEX c, c0, texture[1], 1D; \n\
// MUL c, c, fc; \n\
// LRP result.color, fogFactor.x, c, fogColor; \n\
// END";
*/

#if ! defined(__sgi)
//PFNGLCOLORTABLEEXTPROC glColorTableEXT;
//#include <GL/glu.h>

// extern "C" {
//  void glColorTableEXT (GLenum, GLenum, GLsizei, GLenum, GLenum, const GLvoid *);
// }
#endif

#include <sci_glu.h>


namespace SCIRun {

using std::vector;
using std::cerr;
using std::endl;


GLVolRenState::GLVolRenState(const GLVolumeRenderer* glvr)
  : volren( glvr ),texName(0), reload_(true), 
    newbricks_(false), newcmap_(true)
{
  // Base Class, holds pointer to VolumeRenderer and 
  // common computation
#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
  VolShader = new FragmentProgramARB( ShaderString, false );
#endif

}


void
GLVolRenState::computeView(Ray& ray)
{
  double mvmat[16];
  Transform mat;
  Vector view;
  Point viewPt;
      
  glGetDoublev( GL_MODELVIEW_MATRIX, mvmat);
  /* remember that the glmatrix is stored as
     0  4  8 12
     1  5  9 13
     2  6 10 14
     3  7 11 15 */
  
  // transform the view vector opposite the transform that we draw polys with,
  // so that polys are normal to the view post opengl draw.
  //  GLTexture3DHandle tex = volren->get_tex3d_handle();
  //  Transform field_trans = tex->get_field_transform();

  GLTexture3DHandle tex = volren->get_tex3d_handle();
  Transform field_trans = tex->get_field_transform();

  // this is the world space view direction
  view = Vector(-mvmat[2], -mvmat[6], -mvmat[10]);

  // but this is the view space viewPt
  viewPt = Point(-mvmat[12], -mvmat[13], -mvmat[14]);

  viewPt = field_trans.unproject( viewPt );
  view = field_trans.unproject( view );

  /* set the translation to zero */
  mvmat[12]=mvmat[13] = mvmat[14]=0;
   

  /* The Transform stores it's matrix as
     0  1  2  3
     4  5  6  7
     8  9 10 11
     12 13 14 15

     Because of this order, simply setting the tranform with the glmatrix 
     causes our tranform matrix to be the transpose of the glmatrix
     ( assuming no scaling ) */
  mat.set( mvmat );
    
  /* Since mat is the transpose, we then multiply the view space viewPt
     by the mat to get the world or model space viewPt, which we need
     for calculations */
  viewPt = mat.project( viewPt );
 
  ray =  Ray(viewPt, view);
}

void
GLVolRenState::drawPolys( vector<Polygon *> polys )
{
  double mvmat[16];
  GLTexture3DHandle tex = volren->get_tex3d_handle();
  Transform field_trans = tex->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);

  Point p0, t0;

  unsigned int i;
  int k;
  volren->di()->polycount += polys.size();
  for (i = 0; i < polys.size(); i++) {
    switch (polys[i]->size() ) {
    case 1:
//       t0 = polys[i]->getTexCoord(0);
      p0 = polys[i]->getVertex(0);
      glBegin(GL_POINTS);
  //       glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
//       glTexCoord3f(t0.x(), t0.y(), t0.z());
      glVertex3f(p0.x(), p0.y(), p0.z());
      glEnd();
      break;
    case 2:
            glBegin(GL_LINES);
      for(k =0; k < polys[i]->size(); k++)
      {
//         t0 = polys[i]->getTexCoord(k);
        p0 = polys[i]->getVertex(k);
  //         glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
//         glTexCoord3f(t0.x(), t0.y(), t0.z());
        glVertex3f(p0.x(), p0.y(), p0.z());
      }
      glEnd();
      break;
    case 3:
      {
        Vector n = Cross(Vector((*(polys[i]))[0] - (*polys[i])[1]),
                         Vector((*(polys[i]))[0] - (*polys[i])[2]));
        n.normalize();
        glBegin(GL_TRIANGLES);
        glNormal3f(n.x(), n.y(), n.z());
        for(k =0; k < polys[i]->size(); k++)
        {
//           t0 = polys[i]->getTexCoord(k);
          p0 = polys[i]->getVertex(k);
  //           glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
//           glTexCoord3f(t0.x(), t0.y(), t0.z());
          glVertex3f(p0.x(), p0.y(), p0.z());
        }
	glEnd();
      }
      break;
    case 4:
    case 5:
    case 6:
      {
	int k;
	glBegin(GL_POLYGON);
	Vector n = Cross(Vector((*(polys[i]))[0] - (*polys[i])[1]),
			 Vector((*(polys[i]))[0] - (*polys[i])[2]));
	n.normalize();
	glNormal3f(n.x(), n.y(), n.z());
	for(k =0; k < polys[i]->size(); k++)
	{
	  t0 = polys[i]->getTexCoord(k);
	  p0 = polys[i]->getVertex(k);
// 	  glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
	  glTexCoord3f(t0.x(), t0.y(), t0.z());
	  glVertex3f(p0.x(), p0.y(), p0.z());
	  //            sprintf(s, "3D texture coordinates are ( %f, %f, %f, )\n", t0.x(), t0.y(), t0.z() );
	  // 	cerr<<s;
	  //            sprintf(s, "2D texture coordinates are ( %f, %f )\n", t1_0, t1_1);
	  //            cerr<<s;
	}
	glEnd();
	break;
      }
    }
  }
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void
GLVolRenState::loadColorMap(Brick& brick)
{
  CHECK_OPENGL_ERROR("start of loadColorMap")
  const unsigned char *arr = volren->transfer_functions(brick.level());

#if defined(GL_ARB_fragment_program) && defined(GL_ARB_multitexture)  && defined(__APPLE__)

  glActiveTextureARB(GL_TEXTURE1_ARB);
  {
    glEnable(GL_TEXTURE_1D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if( cmap_texture_ == 0 || newcmap_ ){
      glDeleteTextures(1, &cmap_texture_);
      glGenTextures(1, &cmap_texture_);
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage1D(GL_TEXTURE_1D, 0,
		   GL_RGBA,
		   256, 0,
		   GL_RGBA, GL_UNSIGNED_BYTE,
		   arr);
      newcmap_ = false;
    } else {
      glBindTexture(GL_TEXTURE_1D, cmap_texture_);
    }
    glActiveTexture(GL_TEXTURE0_ARB);
  }
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI ) && defined(__sgi)
  //cerr << "GL_TEXTURE_COLOR_TABLE_SGI defined" << endl;
  glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
               GL_RGBA,
               256, // try larger sizes?
               GL_RGBA,  // need an alpha value...
               GL_UNSIGNED_BYTE, // try shorts...
               arr);
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
  //cerr << "GL_SHARED_TEXTURE_PALETTE_EXT  defined" << endl;

#ifndef HAVE_CHROMIUM
  //cerr << "not HAVE_CHROMIUM" << endl;
  //ASSERT(glColorTableEXT != NULL );
  glColorTable(GL_SHARED_TEXTURE_PALETTE_EXT,
		  GL_RGBA,
		  256, // try larger sizes?
		  GL_RGBA,  // need an alpha value...
		  GL_UNSIGNED_BYTE, // try shorts...
		  arr);
  CHECK_OPENGL_ERROR("After glColorTableEXT")
#endif

#endif
  CHECK_OPENGL_ERROR("end of loadColorMap")
}

void 
GLVolRenState::loadTexture(Brick& brick)
{
  CHECK_OPENGL_ERROR("start of GLVolRenState::loadTexture")
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  glActiveTexture(GL_TEXTURE0_ARB);
//   glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);
#endif
  if( !brick.texName() || reload_ ) {
    if( !brick.texName() ){
      glGenTextures(1, brick.texNameP());
      textureNames.push_back( brick.texName() );
     }
    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
    CHECK_OPENGL_ERROR("After glBindTexture")
    if(volren->interp()){
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      CHECK_OPENGL_ERROR("glTexParameteri GL_LINEAR");
    } else {
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      CHECK_OPENGL_ERROR("glTexParameteri GL_NEAREST")
    }


#if defined( GL_ARB_fragment_program )  && defined(GL_ARB_multitexture)  && defined(__APPLE__)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
//     {
//       int border_color[4] = {0, 0, 0, 0};
//       glTexParameteriv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);
//     }
#else
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S,
		    GL_CLAMP);
    CHECK_OPENGL_ERROR("glTexParameteri GL_TEXTURE_WRAP_S GL_CLAMP")
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T,
		    GL_CLAMP);
    CHECK_OPENGL_ERROR("glTexParameteri GL_TEXTURE_WRAP_T GL_CLAMP")
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT,
		    GL_CLAMP);
    CHECK_OPENGL_ERROR("glTexParameteri GL_TEXTURE_WRAP_R GL_CLAMP")
#endif
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    CHECK_OPENGL_ERROR("After glPixelStorei(GL_UNPACK_ALIGNMENT, 1)")
    
#if defined( GL_ARB_fragment_program ) && defined(GL_ARB_multitexture) && defined(__APPLE__)
    glTexImage3D(GL_TEXTURE_3D, 0,
		 GL_INTENSITY,
		 (brick.texture())->dim1(), 
		 (brick.texture())->dim2(), 
		 (brick.texture())->dim3(),
		 0,
		 GL_LUMINANCE, GL_UNSIGNED_BYTE,
		 &(*(brick.texture()))(0,0,0));
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI ) && defined(__sgi)
    // set up the texture
    //glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0,
    glTexImage3DEXT(GL_TEXTURE_3D, 0,
		    GL_INTENSITY8,
		    (brick.texture())->dim1(), 
		    (brick.texture())->dim2(), 
		    (brick.texture())->dim3(),
		    0,
		    GL_RED, GL_UNSIGNED_BYTE,
		    &(*(brick.texture()))(0,0,0));
    CHECK_OPENGL_ERROR("After glTexImage3D SGI")
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
    glTexImage3D(GL_TEXTURE_3D_EXT, 0,
		 GL_COLOR_INDEX8_EXT,
		 (brick.texture())->dim1(),
		 (brick.texture())->dim2(),
		 (brick.texture())->dim3(),
		 0,
		 GL_COLOR_INDEX, GL_UNSIGNED_BYTE,
		 &(*(brick.texture()))(0,0,0));
    CHECK_OPENGL_ERROR("After glTexImage3D Linux")
#endif
  } else {
    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
  }
}
void 
GLVolRenState::makeTextureMatrix( const Brick& brick)
{
  double splane[4]={0,0,0,0};
  double tplane[4]={0,0,0,0};
  double rplane[4]={0,0,0,0};
  double qplane[4]={0,0,0,1};


  Vector diag;

  
  /* The cube is numbered in the following way 
      
         2________ 6        y
        /|       /|         |  
       / |      / |         |
      /  |     /  |         |
    3/__0|____/7__|4        |_________ x
     |   /    |   /         /
     |  /     |  /         /
     | /      | /         /
    1|/_______|/5        /
                        z  
  */



  diag = brick[7] - brick[0];


  glTexGend(GL_S,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_T,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_R,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);
  glTexGend(GL_Q,GL_TEXTURE_GEN_MODE,GL_OBJECT_LINEAR);

  //  This code is for render overlapping bricks.  The plane equations
  //  for s are  (Nx * Pxmin) + d = aX/2  and
  //  (Nx * Pxmax) + d = 1 - aX/2 where
  //  Nx is the x component of the normal,  Pxmin and Pxmax are the x 
  //  components of the min and max points on the TexCube, and  aX is one
  //  texel width.  Solving for Nx and d we get
  //  Nx = (1 - aX)/(Pxmax - Pxmin) and
  //  d = aX/2 - (Pxmin *(1 - aX))/(Pxmax - Pxmin)

  splane[0] = (1 - brick.aX * (brick.padx + 1))/diag.x();
  splane[3] = brick.aX * 0.5 - (brick[0].x() *
				(1 - brick.aX * (brick.padx+1))/diag.x());
  tplane[1] = (1 - brick.aY * (brick.pady + 1))/diag.y();
  tplane[3] = brick.aY * 0.5 - (brick[0].y() *
				(1 - brick.aY * (brick.pady+1))/diag.y());
  rplane[2] = (1 - brick.aZ * (brick.padz + 1))/diag.z();
  rplane[3] = brick.aZ * 0.5 - (brick[0].z() *
				(1 - brick.aZ * (brick.padz+1))/diag.z());

  
  glTexGendv(GL_S,GL_OBJECT_PLANE,splane);
  glTexGendv(GL_T,GL_OBJECT_PLANE,tplane);
  glTexGendv(GL_R,GL_OBJECT_PLANE,rplane);
  glTexGendv(GL_Q,GL_OBJECT_PLANE,qplane);
}

void 
GLVolRenState::enableTexCoords()
{
  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  glEnable(GL_TEXTURE_GEN_R);
  glEnable(GL_TEXTURE_GEN_Q);
}
void 
GLVolRenState::disableTexCoords()
{
  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
}

void 
GLVolRenState::enableBlend()
{

  glEnable(GL_BLEND);

}
void 
GLVolRenState::disableBlend()
{
  glDisable(GL_BLEND);
}

void
GLVolRenState::drawWireFrame(const Brick& brick)
{
  int i;
  double mvmat[16];
  GLTexture3DHandle tex = volren->get_tex3d_handle();
  Transform field_trans = tex->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);

  glEnable(GL_DEPTH_TEST);
//   double r,g,b;
//   char c;
//   r = drand48();
//   g = drand48();
//   b = drand48();
//   std::cin.get(c);
//   glColor4f(r,g,b,1.0);
  glColor4f(0.8,0.8,0.8,1.0);
  glPushMatrix();
  glBegin(GL_LINES);
  for(i = 0; i < 4; i++){
    glVertex3d(brick[i].x(), brick[i].y(), brick[i].z());
    glVertex3d(brick[i+4].x(), brick[i+4].y(), brick[i+4].z());
  }
  glEnd();

  glBegin(GL_LINE_LOOP);
   glVertex3d(brick[0].x(), brick[0].y(), brick[0].z());
   glVertex3d(brick[1].x(), brick[1].y(), brick[1].z());
   glVertex3d(brick[3].x(), brick[3].y(), brick[3].z());
   glVertex3d(brick[2].x(), brick[2].y(), brick[2].z());
  glEnd();

  glBegin(GL_LINE_LOOP);
   glVertex3d(brick[4].x(), brick[4].y(), brick[4].z());
   glVertex3d(brick[5].x(), brick[5].y(), brick[5].z());
   glVertex3d(brick[7].x(), brick[7].y(), brick[7].z());
   glVertex3d(brick[6].x(), brick[6].y(), brick[6].z());
  glEnd();
  glPopMatrix();
  glDisable(GL_DEPTH_TEST);

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}


#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
FragmentProgramARB::FragmentProgramARB (const char* str, bool isFileName)
  : mId(0), mBuffer(0), mLength(0), mFile(0)
{
  init( str, isFileName);
}

void
FragmentProgramARB::init( const char* str, bool isFileName )
{
  if( isFileName ) {
    if(mFile) delete mFile;
    mFile = new char[strlen(str)];
    strcpy(mFile, str);
    
    FILE *fp;

    if (!(fp = fopen(str,"rb")))
    {
      cerr << "FragProgARB::constructor error: " << str << " could not be read " << endl;
      return;
    }
    
    fseek(fp, 0, SEEK_END);
    mLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if(mBuffer) delete mBuffer;
    mBuffer = new unsigned char[mLength+1];
    
    fread( mBuffer, 1, mLength, fp);
    mBuffer[mLength] = '\0'; // make it a regular C string
    fclose(fp);
  } else {
    mLength = strlen(str);
    mBuffer = new unsigned char[mLength+2];
    strcpy((char*)mBuffer, str);
  }
  
}

FragmentProgramARB::~FragmentProgramARB ()
{
  delete [] mBuffer;
}

bool
FragmentProgramARB::created() {return bool(mId);}

void
FragmentProgramARB::create ()
{
  glGenProgramsARB(1, &mId);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
		     mLength, mBuffer);

  CHECK_OPENGL_ERROR("FragmentProgramARB::create");
}

void
FragmentProgramARB::destroy ()
{
  glDeleteProgramsARB(1, &mId);
  mId = 0;
}

void
FragmentProgramARB::bind ()
{
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
}

void
FragmentProgramARB::release ()
{
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::enable ()
{
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::disable ()
{
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::makeCurrent ()
{
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
}

#endif

} // End namespace SCIRun


