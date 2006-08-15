
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/TextureObj.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geom/ShaderProgramARB.h>

namespace SCIRun {



TextureObj::TextureObj(NrrdDataHandle &nrrd_handle) :
  nrrd_handle_(0),
  width_(-1),
  height_(-1),
  dirty_(true),
  texture_id_(0)
{
  set_color(1.0, 1.0, 1.0, 1.0);
  set_nrrd(nrrd_handle);
}


TextureObj::TextureObj(int components, int x, int y) :
  nrrd_handle_(0),
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
  NrrdDataHandle nrrd_handle = scinew NrrdData();
  nrrdAlloc_nva(nrrd_handle->nrrd_, nrrdTypeUChar, 3, size);
    
  set_nrrd(nrrd_handle);
}

void
TextureObj::set_nrrd(NrrdDataHandle &nrrd_handle) {
  if (!nrrd_handle.get_rep() ||
      !nrrd_handle->nrrd_ ||
       nrrd_handle->nrrd_->dim != 3) 
    throw "TextureObj::set_nrrd(NrrdDataHandle &nrrd_handle): nrrd not valid";
  nrrd_handle_ = nrrd_handle;
  width_  = nrrd_handle_->nrrd_->axis[1].size;
  height_ = nrrd_handle_->nrrd_->axis[2].size;

  pad_to_power_of_2();
}


TextureObj::~TextureObj()
{
  if (glIsTexture(texture_id_)) {
    glDeleteTextures(1, (const GLuint*)&texture_id_);
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  nrrd_handle_ = 0;
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
  //  if (ShaderProgramARB::texture_non_power_of_two()) 
  //    return;
  if (!nrrd_handle_.get_rep() || !nrrd_handle_->nrrd_)
    return;
  if (IsPowerOf2(nrrd_handle_->nrrd_->axis[1].size) &&
      IsPowerOf2(nrrd_handle_->nrrd_->axis[2].size)) 
    return;
  NrrdDataHandle nout_handle = scinew NrrdData();
  ptrdiff_t minp[3] = { 0, 0, 0 };
  ptrdiff_t maxp[3] = { nrrd_handle_->nrrd_->axis[0].size-1, 
			Pow2(nrrd_handle_->nrrd_->axis[1].size)-1, 
			Pow2(nrrd_handle_->nrrd_->axis[2].size)-1 };

  NrrdResampleInfo *info = nrrdResampleInfoNew();
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  info->boundary = nrrdBoundaryBleed;
  info->type = nrrd_handle_->nrrd_->type;
  info->renormalize = AIR_FALSE;

  info->kernel[0] = 0;
  info->kernel[1] = nrrdKernelBox;
  info->kernel[2] = nrrdKernelBox;
  info->samples[0] = nrrd_handle_->nrrd_->axis[0].size;
  info->samples[1] = width_ = Pow2(nrrd_handle_->nrrd_->axis[1].size);
  info->samples[2] = height_ = Pow2(nrrd_handle_->nrrd_->axis[2].size);
  memcpy(info->parm[0], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  memcpy(info->parm[1], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  memcpy(info->parm[2], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));

  Nrrd * nin = nrrd_handle_->nrrd_;

  for (int a = 0; a < nin->dim; ++a) {
    nrrdAxisInfoMinMaxSet(nin, a, nin->axis[a].center ? 
                          nin->axis[a].center : nrrdDefaultCenter);
    
    info->min[a] = nin->axis[a].min;
    info->max[a] = nin->axis[a].max;
  }


//   info->min[0] = 0;
//   info->min[1] = 0;
//   info->min[2] = 0;
//   info->max[0] = nrrd_handle_->nrrd_->axis[0].size;
//   info->max[1] = nrrd_handle_->nrrd_->axis[1].size;
//   info->max[2] = nrrd_handle_->nrrd_->axis[2].size1;


//   nrrd_handle_->nrrd_->axis[0].min = 0;
//   nrrd_handle_->nrrd_->axis[0].max = nrrd_handle_->nrrd_->axis[0].size-1;

//   nrrd_handle_->nrrd_->axis[1].min = 0;
//   nrrd_handle_->nrrd_->axis[1].max = nrrd_handle_->nrrd_->axis[1].size-1;

//   nrrd_handle_->nrrd_->axis[2].min = 0;
//   nrrd_handle_->nrrd_->axis[2].max = nrrd_handle_->nrrd_->axis[2].size-1;



  if (nrrdSpatialResample(nout_handle->nrrd_, nrrd_handle_->nrrd_, info)) {


//   if (nrrdPad_nva(nout_handle->nrrd_, nrrd_handle_->nrrd_,
// 		  minp, maxp, nrrdBoundaryBleed, 1)) {
    char *err = biffGetDone(NRRD);
    string error = string("Trouble resampling: ") + err;
    free (err);
    throw error;
  }
  nrrd_handle_ = nout_handle;
  set_dirty();
}


bool
TextureObj::bind()
{
  if (!nrrd_handle_.get_rep() || !nrrd_handle_->nrrd_) return false;

  const bool bound = glIsTexture(texture_id_);

  if (!bound)
    glGenTextures(1, (GLuint*)&texture_id_);

  glBindTexture(GL_TEXTURE_2D, texture_id_);
  CHECK_OPENGL_ERROR();
  if (bound && !dirty_) return true;
  dirty_ = false;
  Nrrd nrrd = *nrrd_handle_->nrrd_;
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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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
            
  //  glColor4f(0.8, 0.0, 0.5, 1.0);//color_);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

  //  glMatrixMode(GL_TEXTURE);
  //  glPushMatrix();
  //  glLoadIdentity();
  //  glScaled(1.0, -1.0, 1.0);

  if (n == 4) {
    float x_scale = double(width_ )/float(nrrd_handle_->nrrd_->axis[1].size);
    float y_scale = double(height_)/float(nrrd_handle_->nrrd_->axis[2].size);
    glBegin(GL_QUADS);
    
    for (int v = 0; v < 4; ++v) {
      glTexCoord2f(tex_coords[v*2+0] * x_scale, 
                   tex_coords[v*2+1] * y_scale);
      glVertex3dv(&vertices[v](0));
    }
    glEnd();
  }

  //  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
  CHECK_OPENGL_ERROR();
}

}
