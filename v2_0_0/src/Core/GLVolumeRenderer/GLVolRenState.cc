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

#include <GL/gl.h>
#include <Core/GLVolumeRenderer/GLVolRenState.h>
#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Core/GLVolumeRenderer/GLTexture3D.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
using namespace std;

#if ! defined(__sgi)
//PFNGLCOLORTABLEEXTPROC glColorTableEXT;
//#include <GL/glu.h>

// extern "C" {
//  void glColorTableEXT (GLenum, GLenum, GLsizei, GLenum, GLenum, const GLvoid *);
// }
#endif

#include <sci_glu.h>
GLenum errCode;
const GLubyte *errString;

void glCheckForError(const char* message)
{
  if((errCode = glGetError()) != GL_NO_ERROR){
    errString = gluErrorString(errCode);
    cerr<<"OpenGL Error: "<<message<<" "<<(const char*)errString<<endl;
  }
}

namespace SCIRun {

using std::vector;
using std::cerr;
using std::endl;


GLVolRenState::GLVolRenState(const GLVolumeRenderer* glvr)
  : volren( glvr ), texName(0), reload((unsigned char *)1), newbricks_(false)
{
  // Base Class, holds pointer to VolumeRenderer and 
  // common computation
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

  unsigned int i;
  volren->di()->polycount += polys.size();
  for (i = 0; i < polys.size(); i++) {
    switch (polys[i]->size() ) {
    case 1:
      glBegin(GL_POINTS);
      glVertex3f((*(polys[i]))[0].x(),(*(polys[i]))[0].y(),
		 (*(polys[i]))[0].z());
      glEnd();
      break;
    case 2:
      glBegin(GL_LINES);
      glVertex3f((*(polys[i]))[0].x(),(*(polys[i]))[0].y(),
		 (*(polys[i]))[0].z());
      glVertex3f((*(polys[i]))[1].x(), (*(polys[i]))[1].y(),
		 (*(polys[i]))[1].z());
      glEnd();
      break;
    case 3:
      {
	glBegin(GL_TRIANGLES);
	Vector n = Cross(Vector((*(polys[i]))[0] - (*polys[i])[1]),
			 Vector((*(polys[i]))[0] - (*polys[i])[2]));
	n.normalize();
	glNormal3f(n.x(), n.y(), n.z());
	glVertex3f((*(polys[i]))[0].x(),(*(polys[i]))[0].y(),
		   (*(polys[i]))[0].z());
	glVertex3f((*(polys[i]))[1].x(), (*(polys[i]))[1].y(),
		   (*(polys[i]))[1].z());
	glVertex3f((*(polys[i]))[2].x(),(*(polys[i]))[2].y(),
		   (*(polys[i]))[2].z());
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
	  glVertex3f((*(polys[i]))[k].x(),(*(polys[i]))[k].y(),
		     (*(polys[i]))[k].z());
	}
	glEnd();
      }
      break;
    }
  }

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

void
GLVolRenState::loadColorMap(Brick& brick)
{
  const unsigned char *arr = volren->transfer_functions(brick.level());
#ifdef GL_TEXTURE_COLOR_TABLE_SGI
  glColorTable(GL_TEXTURE_COLOR_TABLE_SGI,
               GL_RGBA,
               256, // try larger sizes?
               GL_RGBA,  // need an alpha value...
               GL_UNSIGNED_BYTE, // try shorts...
               arr);
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
  ASSERT(glColorTableEXT != NULL );
  glColorTableEXT(GL_SHARED_TEXTURE_PALETTE_EXT,
	       GL_RGBA,
               256, // try larger sizes?
               GL_RGBA,  // need an alpha value...
               GL_UNSIGNED_BYTE, // try shorts...
               arr);
//   glCheckForError("After glColorTableEXT");
#endif
}

void 
GLVolRenState::loadTexture(Brick& brick)
{
  if( !brick.texName() || reload ) {
    if( !brick.texName() ){
      glGenTextures(1, brick.texNameP());
      textureNames.push_back( brick.texName() );
    }

    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
//      glCheckForError("After glBindTexture");

    if(volren->interp()){
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//        glCheckForError("glTexParameteri GL_LINEAR");
    } else {
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//        glCheckForError("glTexParameteri GL_NEAREST");
    }

    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S,
		    GL_CLAMP);
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_S GL_CLAMP");
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T,
		    GL_CLAMP);
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_T GL_CLAMP");
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT,
		    GL_CLAMP);
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_R GL_CLAMP");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//      glCheckForError("After glPixelStorei(GL_UNPACK_ALIGNMENT, 1)");
    
#ifdef GL_TEXTURE_COLOR_TABLE_SGI 
    // set up the texture
    //glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0,
    glTexImage3D(GL_TEXTURE_3D, 0,
		    GL_INTENSITY8,
		    (brick.texture())->dim1(), 
		    (brick.texture())->dim2(), 
		    (brick.texture())->dim3(),
		    0,
		    GL_RED, GL_UNSIGNED_BYTE,
		    &(*(brick.texture()))(0,0,0));
//      glCheckForError("After glTexImage3D SGI");
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
    glTexImage3D(GL_TEXTURE_3D_EXT, 0,
		    GL_COLOR_INDEX8_EXT,
		    (brick.texture())->dim1(), 
		    (brick.texture())->dim2(), 
		    (brick.texture())->dim3(),
		    0,
		    GL_COLOR_INDEX, GL_UNSIGNED_BYTE,
		    &(*(brick.texture()))(0,0,0));
//      glCheckForError("After glTexImage3D Linux");
#endif
  } else {
    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
  }
  //#endif
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

} // End namespace SCIRun


