#include <sci_defs.h>
#include <Packages/Volume/Core/Geom/TextureRenderer.h>

#if defined( HAVE_GLEW )
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif

#include <Packages/Volume/Core/Datatypes/TypedBrickData.h>
#include <Core/Util/NotFinished.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>
#include <Packages/Volume/Core/Util/Utils.h>
#include <iostream>
using std::cerr;

using namespace Volume;
using Volume::Brick;

int TextureRenderer::r_count_ = 0;

TextureRenderer::TextureRenderer() :
  GeomObj(),
  cmap_texture_(0),
  tex_(0),
  mutex_("TextureRenderer Mutex"),
  cmap_(0),
  cmap_has_changed_(true),
  interp_(true),
  lighting_(0),
  reload_(true)
{
  r_count_++;
  NOT_FINISHED("TextureRenderer::TextureRenderer()");
}

TextureRenderer::TextureRenderer(TextureHandle tex, ColorMapHandle map) :
  GeomObj(),
  cmap_texture_(0),
  tex_(tex),
  mutex_("TextureRenderer Mutex"),
  cmap_(map),
  cmap_has_changed_(true),
  interp_(true),
  lighting_(0),
  reload_(true)
{
  r_count_++;
}

TextureRenderer::TextureRenderer(const TextureRenderer& copy) :
  GeomObj( copy ),
  cmap_texture_(copy.cmap_texture_),
  tex_(copy.tex_),
  mutex_("TextureRenderer Mutex"),
  cmap_(copy.cmap_),
  cmap_has_changed_(copy.cmap_has_changed_),
  interp_(copy.interp_),
  lighting_(copy.lighting_),
  reload_(copy.reload_)
{
   r_count_++;
}

TextureRenderer::~TextureRenderer()
{
  r_count_--;
}


#define TEXTURERENDERER_VERSION 1

void 
TextureRenderer::io(Piostream&)
{
  // Nothing for now...
  NOT_FINISHED("TextureRenderer::io");
}

bool
TextureRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
  NOT_FINISHED("TextureRenderer::saveobj");
  return false;
}

void
TextureRenderer::compute_view( Ray& ray)
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

  Transform field_trans = tex_->get_field_transform();

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
TextureRenderer::load_texture(Brick& brick)
{
#if defined( GL_ARB_fragment_program) && defined(GL_ARB_multitexture) && defined(__APPLE__)
  glActiveTexture(GL_TEXTURE0_ARB);
//   glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glEnable(GL_TEXTURE_3D);
#endif
  if( !brick.texName() || reload_ ) {
    if( !brick.texName() ){
      glGenTextures(1, brick.texNameP());
//       textureNames.push_back( brick.texName() );
     }
    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
    //      glCheckForError("After glBindTexture");
    
    if(interp_){
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//        glCheckForError("glTexParameteri GL_LINEAR");
    } else {
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//        glCheckForError("glTexParameteri GL_NEAREST");
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
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_S GL_CLAMP");
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T,
		    GL_CLAMP);
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_T GL_CLAMP");
    glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT,
		    GL_CLAMP);
//      glCheckForError("glTexParameteri GL_TEXTURE_WRAP_R GL_CLAMP");
#endif
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//      glCheckForError("After glPixelStorei(GL_UNPACK_ALIGNMENT, 1)");
    if( TypedBrickData<unsigned char> *br =
        dynamic_cast<TypedBrickData<unsigned char>*> (brick.data())){
#if defined( GL_ARB_fragment_program ) && defined(GL_ARB_multitexture) && defined(__APPLE__)
      glTexImage3D(GL_TEXTURE_3D, 0,
		   GL_INTENSITY,
		   br->nx(),
		   br->ny(),
		   br->nz(),
		   0,
		   GL_LUMINANCE, GL_UNSIGNED_BYTE,
		   br->texture());
#elif defined( GL_TEXTURE_COLOR_TABLE_SGI ) 
	  // set up the texture
	  //glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0,
      glTexImage3D(GL_TEXTURE_3D, 0,
		   GL_INTENSITY8,
		   br->nx(),
		   br->ny(),
		   br->nz(),
		   0,
		   GL_RED, GL_UNSIGNED_BYTE,
		   br->texture());
      //      glCheckForError("After glTexImage3D SGI");
#elif defined( GL_SHARED_TEXTURE_PALETTE_EXT )
      glTexImage3D(GL_TEXTURE_3D_EXT, 0,
		   GL_COLOR_INDEX8_EXT,
		   br->nx(),
		   br->ny(),
		   br->nz(),
		   0,
		   GL_COLOR_INDEX, GL_UNSIGNED_BYTE,
		   br->texture());
#endif
    }
//      glCheckForError("After glTexImage3D Linux");
  } else {
    glBindTexture(GL_TEXTURE_3D_EXT, brick.texName());
  }
  //#endif
}

void
TextureRenderer::make_texture_matrix( const Brick& brick)
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

  splane[0] = (1 - brick.ax() * (brick.padx() + 1))/diag.x();
  splane[3] = brick.ax() * 0.5 - (brick[0].x() *
				(1 - brick.ax() * (brick.padx()+1))/diag.x());
  tplane[1] = (1 - brick.ay() * (brick.pady() + 1))/diag.y();
  tplane[3] = brick.ay() * 0.5 - (brick[0].y() *
				(1 - brick.ay() * (brick.pady()+1))/diag.y());
  rplane[2] = (1 - brick.az() * (brick.padz() + 1))/diag.z();
  rplane[3] = brick.az() * 0.5 - (brick[0].z() *
				(1 - brick.az() * (brick.padz()+1))/diag.z());

  
  glTexGendv(GL_S,GL_OBJECT_PLANE,splane);
  glTexGendv(GL_T,GL_OBJECT_PLANE,tplane);
  glTexGendv(GL_R,GL_OBJECT_PLANE,rplane);
  glTexGendv(GL_Q,GL_OBJECT_PLANE,qplane);

}

void
TextureRenderer::enable_tex_coords()
{
  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  glEnable(GL_TEXTURE_GEN_R);
  glEnable(GL_TEXTURE_GEN_Q);
}

void 
TextureRenderer::disable_tex_coords()
{
  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
}


void
TextureRenderer::drawPolys( vector<Polygon *> polys )
{
  double mvmat[16];
  TextureHandle tex = tex_;
  Transform field_trans = tex_->get_field_transform();
  // set double array transposed.  Our matricies are stored transposed 
  // from OpenGL matricies.
  field_trans.get_trans(mvmat);
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMultMatrixd(mvmat);

  Point p0, t0;
  unsigned int i,k;
  di_->polycount += polys.size();
  for (i = 0; i < polys.size(); i++) {
    switch (polys[i]->size() ) {
    case 1:
      t0 = polys[i]->getTexCoord(0);
      p0 = polys[i]->getVertex(0);
      glBegin(GL_POINTS);
//       glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
      glTexCoord3f(t0.x(), t0.y(), t0.z());
      glVertex3f(p0.x(), p0.y(), p0.z());
      glEnd();
      break;
    case 2:
            glBegin(GL_LINES);
      for(k =0; k < polys[i]->size(); k++)
      {
        t0 = polys[i]->getTexCoord(k);
        p0 = polys[i]->getVertex(k);
//         glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
        glTexCoord3f(t0.x(), t0.y(), t0.z());
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
          t0 = polys[i]->getTexCoord(k);
          p0 = polys[i]->getVertex(k);
//           glMultiTexCoord3f(GL_TEXTURE0_ARB, t0.x(), t0.y(), t0.z());
          glTexCoord3f(t0.x(), t0.y(), t0.z());
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
