
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/TextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/ShaderProgramARB.h>

namespace SCIRun {



TextureObj::TextureObj(NrrdDataHandle &nrrd) :
  nrrd_(0),
  width_(-1),
  height_(-1),
  dirty_(true),
  texture_id_(0)
{
  set_color(1.0, 1.0, 1.0, 1.0);
  set_nrrd(nrrd);
}


TextureObj::TextureObj(int components, int x, int y) :
  nrrd_(0),
  width_(-1),
  height_(-1),
  dirty_(true),
  texture_id_(0)
{
  set_color(1.0, 1.0, 1.0, 1.0);

  size_t size[NRRD_DIM_MAX];
  size[0] = components;
  size[1] = x;
  size[2] = y;
  NrrdDataHandle nrrd = scinew NrrdData();
  nrrdAlloc_nva(nrrd->nrrd, nrrdTypeUChar, 3, size);
    
  set_nrrd(nrrd);
}

void
TextureObj::set_nrrd(NrrdDataHandle &nrrd) {
  if (!nrrd.get_rep() || !nrrd->nrrd || nrrd->nrrd->dim != 3) 
    throw "TextureObj::set_nrrd(NrrdDataHandle &nrrd): nrrd not valid";
  nrrd_ = nrrd;
  width_ = nrrd_->nrrd->axis[1].size;
  height_ = nrrd_->nrrd->axis[2].size;

  pad_to_power_of_2();
}


TextureObj::~TextureObj()
{
  if (glIsTexture(texture_id_)) {
    glDeleteTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  nrrd_ = 0;
}


void
TextureObj::set_color(float r, float g, float b, float a)
{
  color_[0] = r;
  color_[1] = g;
  color_[2] = b;
  color_[3] = a;
}

void
TextureObj::set_color(float rgba[4])
{
  set_color(rgba[0], rgba[1], rgba[2], rgba[3]);
}



void
TextureObj::pad_to_power_of_2()
{
  if (ShaderProgramARB::texture_non_power_of_two()) 
    return;
  if (!nrrd_.get_rep() || !nrrd_->nrrd)
    return;
  if (IsPowerOf2(nrrd_->nrrd->axis[1].size) &&
      IsPowerOf2(nrrd_->nrrd->axis[2].size)) 
    return;
  NrrdDataHandle nout = scinew NrrdData();
  ptrdiff_t minp[3] = { 0, 0, 0 };
  ptrdiff_t maxp[3] = { nrrd_->nrrd->axis[0].size-1, 
			Pow2(nrrd_->nrrd->axis[1].size)-1, 
			Pow2(nrrd_->nrrd->axis[2].size)-1 };

  if (nrrdPad_nva(nout->nrrd, nrrd_->nrrd, minp, maxp, nrrdBoundaryBleed, 1)) {
    char *err = biffGetDone(NRRD);
    string error = string("Trouble resampling: ") + err;
    free (err);
    throw error;
  }
  nrrd_ = nout;
  set_dirty();
}


bool
TextureObj::bind()
{
  if (!nrrd_.get_rep() || !nrrd_->nrrd) return false;

  const bool bound = glIsTexture(texture_id_);

  if (!bound)
    glGenTextures(1, &texture_id_);

  glBindTexture(GL_TEXTURE_2D, texture_id_);
  CHECK_OPENGL_ERROR();
  if (bound && !dirty_) return true;
  dirty_ = false;
  Nrrd nrrd = *nrrd_->nrrd;
  int prim = 1;
  GLenum pixtype;
  if (nrrd.axis[0].size == 1) 
    pixtype = GL_ALPHA;  
  else if (nrrd.axis[0].size == 2) 
    pixtype = GL_LUMINANCE_ALPHA;
  else if (nrrd.axis[0].size == 3) 
    pixtype = GL_RGB;
  else if (nrrd.axis[0].size == 4) 
    pixtype = GL_RGBA;
  else {
    prim = 0;
    pixtype = GL_ALPHA;
  }
  GLenum type = 0;
  switch (nrrd.type) {
  case nrrdTypeChar: type = GL_BYTE; break;
  case nrrdTypeUChar: type = GL_UNSIGNED_BYTE; break;
  case nrrdTypeShort: type = GL_SHORT; break;
  case nrrdTypeUShort: type = GL_UNSIGNED_SHORT; break;	
  case nrrdTypeInt: type = GL_INT; break;
  case nrrdTypeUInt: type = GL_UNSIGNED_INT; break;
  case nrrdTypeFloat: type = GL_FLOAT; break;
  default: throw string("Cant bind nrrd"); break;
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  CHECK_OPENGL_ERROR();

  glTexImage2D(GL_TEXTURE_2D, 0, pixtype,
	       nrrd.axis[prim].size, nrrd.axis[prim+1].size, 
	       0, pixtype, type, nrrd.data);
  CHECK_OPENGL_ERROR();
  return true;
}



//   glDisable(GL_CULL_FACE);
//   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//   glShadeModel(GL_FLAT);
//   glEnable(GL_BLEND);
//   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//   CHECK_OPENGL_ERROR();


void
TextureObj::draw(int n, Point *vertices, float *tex_coords)
{
  ASSERT(n == 4);
  if (bind()) {
    glEnable(GL_TEXTURE_2D);
    CHECK_OPENGL_ERROR();
  } else {
    glDisable(GL_TEXTURE_2D);
    CHECK_OPENGL_ERROR();
    return;
  }
  glEnable(GL_TEXTURE_2D);
  glColor4fv(color_);
            
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  if (n == 4) {
    float x_scale = double(width_)/float(nrrd_->nrrd->axis[1].size);
    float y_scale = double(height_)/float(nrrd_->nrrd->axis[2].size);
    glBegin(GL_QUADS);
    
    for (int v = 0; v < 4; ++v) {
      glTexCoord2f(tex_coords[v*2+0] * x_scale, 
                   tex_coords[v*2+1] * y_scale);
      glVertex3dv(&vertices[v](0));
    }
    glEnd();
  }

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();
}

}
