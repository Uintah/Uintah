//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ParticleFlowRenderer.cc
//    Author : Kurt Zimmerman
//    Date   : March 1, 2006


#include <sci_glu.h>

#include <Packages/Uintah/Dataflow/Modules/Visualization/ParticleFlowRenderer.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>


#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using std::cerr;

#if !defined(HAVE_GLEW)

#  if !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES)
      extern "C" void ( * glXGetProcAddressARB (const GLubyte *procName)) (void);
#  endif /* !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES) */
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#    include <stdlib.h>
#    include <string.h>

     static void *NSGLGetProcAddress (const GLubyte *name)
     {
       NSSymbol symbol;
       char *symbolName;
       /* prepend a '_' for the Unix C symbol mangling convention */
       symbolName = (char*)malloc(strlen((const char *)name) + 2);
       strcpy(symbolName+1, (const char *)name);
       symbolName[0] = '_';
       symbol = NULL;
       if (NSIsSymbolNameDefined(symbolName))
         symbol = NSLookupAndBindSymbol(symbolName);
       free(symbolName);
       return symbol ? NSAddressOfSymbol(symbol) : NULL;
     }
#    define getProcAddress(x) (NSGLGetProcAddress((const GLubyte*)x))
#  else
#    ifdef _WIN32
#      define getProcAddress(x) (wglGetProcAddress((LPCSTR) x))
#    else
#      define getProcAddress(x) ((*glXGetProcAddressARB)((const GLubyte*)x))
#    endif /* _WIN32 */
#  endif /* APPLE */

typedef void (GLAPIENTRY * PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (GLAPIENTRY * PFNGLBINDATTRIBLOCATIONPROC) (GLuint program, GLuint index, const GLchar *name);
typedef void (GLAPIENTRY * PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef GLuint (GLAPIENTRY * PFNGLCREATEPROGRAMPROC) (void);
typedef GLuint (GLAPIENTRY * PFNGLCREATESHADERPROC) (GLenum type);
// typedef void (GLAPIENTRY * PFNGLDELETEPROGRAMPROC) (GLuint program);
// typedef void (GLAPIENTRY * PFNGLDELETESHADERPROC) (GLuint shader);
// typedef void (GLAPIENTRY * PFNGLDETACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (GLAPIENTRY * PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
// typedef void (GLAPIENTRY * PFNGLGETACTIVEATTRIBPROC) (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
// typedef void (GLAPIENTRY * PFNGLGETACTIVEUNIFORMPROC) (GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
// typedef void (GLAPIENTRY * PFNGLGETATTACHEDSHADERSPROC) (GLuint program, GLsizei maxCount, GLsizei *count, GLuint *obj);
// typedef GLint (GLAPIENTRY * PFNGLGETATTRIBLOCATIONPROC) (GLuint program, const GLchar *name);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (GLAPIENTRY * PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
// typedef void (GLAPIENTRY * PFNGLGETSHADERSOURCEPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source);
typedef GLint (GLAPIENTRY * PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLchar *name);
// typedef void (GLAPIENTRY * PFNGLGETUNIFORMFVPROC) (GLuint program, GLint location, GLfloat *params);
// typedef void (GLAPIENTRY * PFNGLGETUNIFORMIVPROC) (GLuint program, GLint location, GLint *params);
// typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBDVPROC) (GLuint index, GLenum pname, GLdouble *params);
// typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBFVPROC) (GLuint index, GLenum pname, GLfloat *params);
// typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBIVPROC) (GLuint index, GLenum pname, GLint *params);
// typedef void (GLAPIENTRY * PFNGLGETVERTEXATTRIBPOINTERVPROC) (GLuint index, GLenum pname, GLvoid* *pointer);
// typedef void (GLAPIENTRY * PFNGLGETUNIFORMIVPROC) (GLuint program, GLint location, GLint *params);
// typedef GLboolean (GLAPIENTRY * PFNGLISPROGRAMPROC) (GLuint program);
// typedef GLboolean (GLAPIENTRY * PFNGLISSHADERPROC) (GLuint shader);
typedef void (GLAPIENTRY * PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (GLAPIENTRY * PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const GLchar* *string, const GLint *length);
typedef void (GLAPIENTRY * PFNGLUSEPROGRAMPROC) (GLuint program);
typedef void (GLAPIENTRY * PFNGLUNIFORM1FPROC) (GLint location, GLfloat v0);
// typedef void (GLAPIENTRY * PFNGLUNIFORM2FPROC) (GLint location, GLfloat v0, GLfloat v1);
// typedef void (GLAPIENTRY * PFNGLUNIFORM3FPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef void (GLAPIENTRY * PFNGLUNIFORM4FPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
// typedef void (GLAPIENTRY * PFNGLUNIFORM1FVPROC) (GLint location, GLsizei count, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM2FVPROC) (GLint location, GLsizei count, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM3FVPROC) (GLint location, GLsizei count, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM4FVPROC) (GLint location, GLsizei count, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM1IVPROC) (GLint location, GLsizei count, const GLint *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM2IVPROC) (GLint location, GLsizei count, const GLint *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM3IVPROC) (GLint location, GLsizei count, const GLint *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORM4IVPROC) (GLint location, GLsizei count, const GLint *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX2FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX3FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLUNIFORMMATRIX4FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
// typedef void (GLAPIENTRY * PFNGLVALIDATEPROGRAMPROC) (GLuint program);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DPROC) (GLuint index, GLdouble x);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1DVPROC) (GLuint index, const GLdouble *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FPROC) (GLuint index, GLfloat x);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1FVPROC) (GLuint index, const GLfloat *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SPROC) (GLuint index, GLshort x);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB1SVPROC) (GLuint index, const GLshort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DPROC) (GLuint index, GLdouble x, GLdouble y);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2DVPROC) (GLuint index, const GLdouble *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FPROC) (GLuint index, GLfloat x, GLfloat y);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2FVPROC) (GLuint index, const GLfloat *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SPROC) (GLuint index, GLshort x, GLshort y);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB2SVPROC) (GLuint index, const GLshort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3DVPROC) (GLuint index, const GLdouble *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3FVPROC) (GLuint index, const GLfloat *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SPROC) (GLuint index, GLshort x, GLshort y, GLshort z);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB3SVPROC) (GLuint index, const GLshort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NBVPROC) (GLuint index, const GLbyte *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NIVPROC) (GLuint index, const GLint *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NSVPROC) (GLuint index, const GLshort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUBPROC) (GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUBVPROC) (GLuint index, const GLubyte *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUIVPROC) (GLuint index, const GLuint *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4NUSVPROC) (GLuint index, const GLushort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4BVPROC) (GLuint index, const GLbyte *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DPROC) (GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4DVPROC) (GLuint index, const GLdouble *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FPROC) (GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4FVPROC) (GLuint index, const GLfloat *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4IVPROC) (GLuint index, const GLint *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SPROC) (GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4SVPROC) (GLuint index, const GLshort *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UBVPROC) (GLuint index, const GLubyte *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4UIVPROC) (GLuint index, const GLuint *v);
// typedef void (GLAPIENTRY * PFNGLVERTEXATTRIB4USVPROC) (GLuint index, const GLushort *v);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBPOINTERVPROC) (GLuint, GLint, GLenum, GLboolean, GLsizei, const GLvoid *);

static PFNGLATTACHSHADERPROC glAttachShader = 0;
static PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation = 0;
static PFNGLCOMPILESHADERPROC glCompileShader = 0;
static PFNGLCREATEPROGRAMPROC glCreateProgram = 0;
static PFNGLCREATESHADERPROC glCreateShader = 0;
// static PFNGLDELETEPROGRAMPROC glDeleteProgram = 0;
// static PFNGLDELETESHADERPROC glDeleteShader = 0;
// static PFNGLDETACHSHADERPROC glDetachShader = 0;
static PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray = 0;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = 0;
// static PFNGLGETACTIVEATTRIBPROC glGetActiveAttrib = 0;
// static PFNGLGETACTIVEUNIFORMPROC glGetActiveUniform = 0;
// static PFNGLGETATTACHEDSHADERSPROC glGetAttachedShaders = 0;
// static PFNGLGETATTRIBLOCATIONPROC glGetAttribLocation = 0;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = 0;
static PFNGLGETPROGRAMIVPROC glGetProgramiv = 0;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = 0;
static PFNGLGETSHADERIVPROC glGetShaderiv = 0;
// static PFNGLGETSHADERSOURCEPROC glGetShaderSource = 0;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = 0;
// static PFNGLGETUNIFORMFVPROC glGetUniformfv = 0;
// static PFNGLGETUNIFORMIVPROC glGetUniformiv = 0;
// static PFNGLGETVERTEXATTRIBDVPROC glGetVertexAttribdv = 0; 
// static PFNGLGETVERTEXATTRIBFVPROC glGetVertexAttribfv = 0; 
// static PFNGLGETVERTEXATTRIBIVPROC glGetVertexAttribiv = 0;
// static PFNGLGETVERTEXATTRIBPOINTERVPROC glGetVertexAttribPointerv = 0; 
// static PFNGLISPROGRAMPROC glIsProgram = 0;
// static PFNGLISSHADERPROC glIsShader = 0;
static PFNGLLINKPROGRAMPROC glLinkProgram = 0;
static PFNGLSHADERSOURCEPROC glShaderSource = 0;
static PFNGLUSEPROGRAMPROC glUseProgram = 0;
static PFNGLUNIFORM1FPROC glUniform1f = 0;
// static PFNGLUNIFORM2FPROC glUniform2f = 0;
// static PFNGLUNIFORM3FPROC glUniform3f = 0;
static PFNGLUNIFORM4FPROC glUniform4f = 0;
// static PFNGLUNIFORM1IPROC glUniform1i = 0;
// static PFNGLUNIFORM2IPROC glUniform2i = 0;
// static PFNGLUNIFORM3IPROC glUniform3i = 0;
// static PFNGLUNIFORM4IPROC glUniform4i = 0;
// static PFNGLUNIFORM1FVPROC glUniform1fv = 0;
// static PFNGLUNIFORM2FVPROC glUniform2fv = 0;
// static PFNGLUNIFORM3FVPROC glUniform3fv = 0;
// static PFNGLUNIFORM4FVPROC glUniform4fv = 0;
// static PFNGLUNIFORM1IVPROC glUniform1iv = 0;
// static PFNGLUNIFORM2IVPROC glUniform2iv = 0;
// static PFNGLUNIFORM3IVPROC glUniform3iv = 0;
// static PFNGLUNIFORM4IVPROC glUniform4iv = 0;
// static PFNGLUNIFORMMATRIX2FVPROC glUniformMatrix2fv = 0;
// static PFNGLUNIFORMMATRIX3FVPROC glUniformMatrix3fv = 0;
// static PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = 0;
// static PFNGLVALIDATEPROGRAMPROC glValidateProgram = 0;
// static PFNGLVERTEXATTRIB1DPROC glVertexAttrib1d = 0;
// static PFNGLVERTEXATTRIB1DVPROC glVertexAttrib1dv = 0;
// static PFNGLVERTEXATTRIB1FPROC glVertexAttrib1f = 0;
// static PFNGLVERTEXATTRIB1FVPROC glVertexAttrib1fv = 0;
// static PFNGLVERTEXATTRIB1SPROC glVertexAttrib1s = 0;
// static PFNGLVERTEXATTRIB1SVPROC glVertexAttrib1sv = 0;
// static PFNGLVERTEXATTRIB2DPROC glVertexAttrib2d = 0;
// static PFNGLVERTEXATTRIB2DVPROC glVertexAttrib2dv = 0;
// static PFNGLVERTEXATTRIB2FPROC glVertexAttrib2f = 0;
// static PFNGLVERTEXATTRIB2FVPROC glVertexAttrib2fv = 0;
// static PFNGLVERTEXATTRIB2SPROC glVertexAttrib2s = 0;
// static PFNGLVERTEXATTRIB2SVPROC glVertexAttrib2sv = 0;
// static PFNGLVERTEXATTRIB3DPROC glVertexAttrib3d = 0;
// static PFNGLVERTEXATTRIB3DVPROC glVertexAttrib3dv = 0;
// static PFNGLVERTEXATTRIB3FPROC glVertexAttrib3f = 0;
// static PFNGLVERTEXATTRIB3FVPROC glVertexAttrib3fv = 0;
// static PFNGLVERTEXATTRIB3SPROC glVertexAttrib3s = 0;
// static PFNGLVERTEXATTRIB3SVPROC glVertexAttrib3sv = 0;
// static PFNGLVERTEXATTRIB4NBVPROC glVertexAttrib4Nbv = 0;
// static PFNGLVERTEXATTRIB4NIVPROC glVertexAttrib4Niv = 0;
// static PFNGLVERTEXATTRIB4NSVPROC glVertexAttrib4Nsv = 0;
// static PFNGLVERTEXATTRIB4NUBPROC glVertexAttrib4Nub = 0;
// static PFNGLVERTEXATTRIB4NUBVPROC glVertexAttrib4Nubv = 0;
// static PFNGLVERTEXATTRIB4NUIVPROC glVertexAttrib4Nuiv = 0;
// static PFNGLVERTEXATTRIB4NUSVPROC glVertexAttrib4Nusv = 0;
// static PFNGLVERTEXATTRIB4BVPROC glVertexAttrib4bv = 0;
// static PFNGLVERTEXATTRIB4DPROC glVertexAttrib4d = 0;
// static PFNGLVERTEXATTRIB4DVPROC glVertexAttrib4dv = 0;
// static PFNGLVERTEXATTRIB4FPROC glVertexAttrib4f = 0;
// static PFNGLVERTEXATTRIB4FVPROC glVertexAttrib4fv = 0;
// static PFNGLVERTEXATTRIB4IVPROC glVertexAttrib4iv = 0;
// static PFNGLVERTEXATTRIB4SPROC glVertexAttrib4s = 0;
// static PFNGLVERTEXATTRIB4SVPROC glVertexAttrib4sv = 0;
// static PFNGLVERTEXATTRIB4UBVPROC glVertexAttrib4ubv = 0;
// static PFNGLVERTEXATTRIB4UIVPROC glVertexAttrib4uiv = 0;
// static PFNGLVERTEXATTRIB4USVPROC glVertexAttrib4usv = 0;
static PFNGLVERTEXATTRIBPOINTERVPROC glVertexAttribPointer = 0;
#endif

int printOglError(char *file, int line);
#define printOpenGLError() printOglError(__FILE__, __LINE__)
int printOglError(char *file, int line)
{
    //
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    //
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        printf("glError 0x%x file %s @ %d: %s\n", 
               glErr, file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}


float fXDiff = 206;
float fYDiff = 16;
float fZDiff = 10;
float fScale = 0.25;

// static members
bool ParticleFlowRenderer::functions_initialized_(false);
bool ParticleFlowRenderer::shader_functions_initialized_(false);

ParticleFlowRenderer::ParticleFlowRenderer() :
  GeomObj(),
  initialized_(false),
  cmap_dirty_(true),
  flow_tex_dirty_(true),
  animating_(false),
  reset_(true),
  time_(0.0),
  time_increment_(0.002f),
  particle_time_(0.0),
  flow_tex_(0),
  vfield_(0),
  cmap_h_(0),
  fh_(0),
  di_(0),
  shader_(0),
  array_width_(1),
  array_height_(1),
  verts_(0),
  colors_(0),
  velocities_(0),
  start_times_(0)
{
}

ParticleFlowRenderer::ParticleFlowRenderer(const ParticleFlowRenderer& copy):
  initialized_(copy.initialized_),
  cmap_dirty_(copy.cmap_dirty_),
  flow_tex_dirty_(copy.flow_tex_dirty_),
  animating_(copy.animating_),
  reset_(copy.reset_),
  time_( copy.time_),
  time_increment_(copy.time_increment_),
  particle_time_(copy.particle_time_),
  flow_tex_(copy.flow_tex_),
  vfield_(copy.vfield_),
  cmap_h_( copy.cmap_h_),
  fh_(copy.fh_),
  di_(copy.di_),
  shader_(copy.shader_),
  array_width_(copy.array_width_),
  array_height_(copy.array_height_),
  verts_(copy.verts_),
  colors_(copy.colors_),
  velocities_(copy.velocities_),
  start_times_(copy.start_times_)
{}

ParticleFlowRenderer::~ParticleFlowRenderer()
{
  cmap_h_ = 0;
  fh_ = 0;
  delete shader_;
}

GeomObj*
ParticleFlowRenderer::clone()
{
  return new ParticleFlowRenderer(*this);
}

void 
ParticleFlowRenderer::update_colormap( ColorMapHandle cmap )
{
  cmap_h_ = cmap;
  if(cmap_h_ != 0 && fh_ != 0 ){
    initialized_ = true;
  }
}

void 
ParticleFlowRenderer::update_vector_field( FieldHandle vfh, bool normalize )
{
  fh_ = vfh;
  if(cmap_h_ != 0 && fh_ != 0 ){
    initialized_ = true;
  }

  // we have already determined in ParticleFlow.cc that we have a 
  // vector field and a LatVolMesh, so cast the data and 
  // build a 3D texture.

  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type LVMeshHandle;
  typedef GenericField<LVMesh, ConstantBasis<Vector>,
                       FData3d<Vector, LVMesh> > LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<Vector>,
                       FData3d<Vector, LVMesh> > LVFieldLB;
  
  if( vfh->basis_order() == 0 ){
    LVFieldCB *fld = (LVFieldCB *)vfh.get_rep();
    LVMesh *mesh = fld->get_typed_mesh().get_rep();
    LVMesh::Cell::iterator it; mesh->begin(it);
    LVMesh::Cell::iterator it_end; mesh->end(it_end);

    nx_ = mesh->get_ni(); ny_ = mesh->get_nj(); nz_ = mesh->get_nk();
   
    if( vfield_ != 0 ) delete [] vfield_;
    vfield_ = scinew GLfloat[nx_ * ny_ * nz_ * 3];
    int i = 0;
    for( ; it != it_end; ++it ){
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).x();
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).y();
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).z();
    }
  } else {  // basis better = 1
    LVFieldLB *fld = (LVFieldLB *)vfh.get_rep();
    LVMesh *mesh = fld->get_typed_mesh().get_rep();
    LVMesh::Node::iterator it; mesh->begin(it);
    LVMesh::Node::iterator it_end; mesh->end(it_end);

    nx_ = mesh->get_ni(); ny_ = mesh->get_nj(); nz_ = mesh->get_nk();
   
    if( vfield_ != 0 ) delete [] vfield_;
    vfield_ = scinew GLfloat[nx_ * ny_ * nz_ * 3];
    int i = 0;
    for( ; it != it_end; ++it ){
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).x();
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).y();
      vfield_[i++] = (GLfloat)(fld->fdata()[*it]).z();
    }
    flow_tex_dirty_ = true;
  }
}

void 
ParticleFlowRenderer::update_time( double time )
{
  time_ = time;
}
 
#ifdef SCI_OPENGL
void 
ParticleFlowRenderer::draw(DrawInfoOpenGL* di, Material* mat, double /* time */)
{
  if(!pre_draw(di, mat, 0)) return;
  di_ = di;
  {
    if( flow_tex_dirty_ ){
      reload_flow_texture();
    }
    Transform  tform;
    fh_->mesh()->get_canonical_transform(tform);
    double mvmat[16];
    tform.get_trans(mvmat);
    glMatrixMode(GL_MODELVIEW);
//     glPushMatrix();
//     glMultMatrixd(mvmat);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glEnable(GL_DEPTH_TEST);
    GLboolean lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    glColor4f(1.0,1.0,1.0,1.0);
    draw_flow_outline();
    if(lighting) glEnable(GL_LIGHTING);
    drawPoints();

//     glPopMatrix();
  }
  di = 0;
}


void ParticleFlowRenderer::draw_flow_outline()
{
  BBox bb =  fh_->mesh()->get_bounding_box();
  
  Point p0(bb.min());
  Point p7(bb.max());
 /* The cube is numbered in the following way
  
       2________6        y
      /|        |        |
     / |       /|        |
    /  |      / |        |
   /   0_____/__4        |
  3---------7   /        |_________ x
  |  /      |  /         /
  | /       | /         /
  |/        |/         /
  1_________5         /
                     z 
 *********************************************/
  glBegin(GL_LINES); 
  {
    glVertex3d(p0.x(), p0.y(), p0.z()); // p0
    glVertex3d(p0.x(), p0.y(), p7.z()); // p1
    glVertex3d(p0.x(), p7.y(), p0.z()); // p2
    glVertex3d(p0.x(), p7.y(), p7.z()); // p3
    glVertex3d(p7.x(), p0.y(), p0.z()); // p4   
    glVertex3d(p7.x(), p0.y(), p7.z()); // p5
    glVertex3d(p7.x(), p7.y(), p0.z()); // p6
    glVertex3d(p7.x(), p7.y(), p7.z()); // p7
  } 
  glEnd();

  glBegin(GL_LINE_LOOP);
  {
    glVertex3d(p0.x(), p0.y(), p0.z()); // p0
    glVertex3d(p0.x(), p7.y(), p0.z()); // p2
    glVertex3d(p7.x(), p7.y(), p0.z()); // p6
    glVertex3d(p7.x(), p0.y(), p0.z()); // p4   
  }
  glEnd();

  glBegin(GL_LINE_LOOP);
  {
    glVertex3d(p0.x(), p0.y(), p7.z()); // p1
    glVertex3d(p0.x(), p7.y(), p7.z()); // p3
    glVertex3d(p7.x(), p7.y(), p7.z()); // p7
    glVertex3d(p7.x(), p0.y(), p7.z()); // p5    
  }
  glEnd();
}
void ParticleFlowRenderer::reload_flow_texture()
{
  if( !flow_tex_ || flow_tex_dirty_ ){
    if( glIsTexture(flow_tex_)){
      glDeleteTextures(1, &flow_tex_);
    }

    glActiveTexture(GL_TEXTURE2_ARB);
    glEnable(GL_TEXTURE_3D);
    glGenTextures(1, &flow_tex_);
    glBindTexture(GL_TEXTURE_3D, flow_tex_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, nx_, ny_, nz_, 0,
                 GL_RGB, GL_FLOAT, vfield_); 

    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE0_ARB);
    
    flow_tex_dirty_ = false;
  }
}

void  
ParticleFlowRenderer::createPoints(GLint w, GLint h)
{
	GLfloat *vptr, *cptr, *velptr, *stptr;
	GLfloat i, j;

	if (verts_ != 0) 
          delete verts_;
        if (colors_ != 0)
          delete colors_;
        if (velocities_ != 0)
          delete velocities_;
        if (start_times_ != 0)
          delete start_times_;

	verts_  = scinew GLfloat[w * h * 3 * sizeof(float)];
	colors_ = scinew GLfloat[w * h * 3 * sizeof(float)];
	velocities_ = scinew GLfloat[w * h * 3 * sizeof(float)];
	start_times_ = scinew GLfloat[w * h * sizeof(float)];

	vptr = verts_;
	cptr = colors_;
	velptr = velocities_;
	stptr  = start_times_;

	for (i = 0.5 / w - 0.5; i < 0.5; i = i + 1.0/w)
		for (j = 0.5 / h - 0.5; j < 0.5; j = j + 1.0/h)
		{
			*vptr       = i;
			*(vptr + 1) = 0.0;
			*(vptr + 2) = j;
			vptr += 3;

			*cptr       = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			*(cptr + 1) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			*(cptr + 2) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			cptr += 3;

			*velptr       = (((float) rand() / RAND_MAX)) + 3.0;
			*(velptr + 1) =  ((float) rand() / RAND_MAX) * 10.0;
			*(velptr + 2) = (((float) rand() / RAND_MAX)) + 3.0;
			velptr += 3;

			*stptr = ((float) rand() / RAND_MAX) * 10.0;
			stptr++;
		}

	array_width_  = w;
	array_height_ = h;
}


void 
ParticleFlowRenderer::updateAnim()
{
  int location;
  location = shader_->getUniLoc(shader_->ProgramObject, "Time"); 

  particle_time_ += time_increment_;
  if (particle_time_ > 15.0)
    particle_time_ = 0.0;

  glUniform1f(location, particle_time_);

  printOpenGLError();  // Check for OpenGL errors
}

void  
ParticleFlowRenderer::drawPoints()
{	

  if( !shader_functions_initialized_){
    if( ParticleFlowShader::build_shader_functions() ){
      shader_functions_initialized_ = true;
      createPoints(100,100);
    } else {
      cerr<<"shader functions not initialized\n";
      return; // do nothing
    }
  }

  GLchar *vertex_shader_source, *fragment_shader_source;

  if( shader_ == 0 ){
    shader_ = scinew ParticleFlowShader();
    shader_->readShaderSource("particle",&vertex_shader_source, 
                                         &fragment_shader_source);
    int success = shader_->installParticleShaders(vertex_shader_source,
                                                  fragment_shader_source);
    if( !success ){
      cerr<<"Shader installation failed\n";
      delete shader_;
      shader_ = 0;
      return;
    }
  }

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   //   glLoadIdentity();
   
   // set up transform for shader
   GLdouble mat[16];
   shader_trans_.get_trans( mat );
   for(int i = 0; i < 4; i++) {
     for( int j = 0; j< 4; j++) {
       cerr<< mat[i*4 + j]<< " ";
     }
     cerr<<"\n";
   }
   cerr<<"\n";
   glMultMatrixd( mat );

   glUseProgram(shader_->ProgramObject);

  if(animating_) updateAnim();

  

//   glLoadIdentity();
//   glTranslatef(0.0, 0.0, -5.0);
  
//   glRotatef(fYDiff, 1,0,0);
//   glRotatef(fXDiff, 0,1,0);
//   glRotatef(fZDiff, 0,0,1);
//   glScalef(fScale, fScale, fScale);


  GLboolean depth;
  glGetBooleanv(GL_DEPTH_TEST, &depth);
  glDepthFunc(GL_LESS);

  if( !depth ){
    glEnable(GL_DEPTH_TEST);
  }


  
  glPointSize(2.0);

  glVertexPointer(3, GL_FLOAT, 0, verts_);
  glColorPointer(3, GL_FLOAT, 0, colors_);
  glVertexAttribPointer(VELOCITY_ARRAY,  3, GL_FLOAT, GL_FALSE, 0, velocities_);
  glVertexAttribPointer(START_TIME_ARRAY, 1, GL_FLOAT, GL_FALSE, 0, start_times_);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableVertexAttribArray(VELOCITY_ARRAY);
  glEnableVertexAttribArray(START_TIME_ARRAY);

  glDrawArrays(GL_POINTS, 0, array_width_ * array_height_);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableVertexAttribArray(VELOCITY_ARRAY);
  glDisableVertexAttribArray(START_TIME_ARRAY);
  
  if( !depth ){
    glDisable(GL_DEPTH_TEST);
  }
  
  glUseProgram(0);
  glPopMatrix();
}

void
ParticleFlowRenderer::update_transform(const Point& c, const Point& r, 
                                       const Point& d)
{

  Transform t;
  Point unused;
  shader_trans_.load_identity();
  shader_trans_.pre_scale( Vector( (r - c).length(),
                                   1.0,
                                   (d - c).length()));

  t.load_frame(unused, Vector(-2,0,0), Vector(0,1,0), Vector(0,0,2));
  shader_trans_.pre_trans(t);
  shader_trans_.pre_translate( c.asVector() );
  
}

bool ParticleFlowShader::shader_functions_built_ = false;

bool
ParticleFlowShader::build_shader_functions()
{
  if(!shader_functions_built_){
    shader_functions_built_ = 
      ((glAttachShader = (PFNGLATTACHSHADERPROC)
        getProcAddress("glAttachShader"))&&
       (glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC)
        getProcAddress("glBindAttribLocation"))&&
       (glCompileShader = (PFNGLCOMPILESHADERPROC)
        getProcAddress("glCompileShader"))&&
       (glCreateProgram = (PFNGLCREATEPROGRAMPROC)
        getProcAddress("glCreateProgram"))&&
       (glCreateShader = (PFNGLCREATESHADERPROC)
        getProcAddress("glCreateShader"))&&
//        (glDeleteProgram = (PFNGLDELETEPROGRAMPROC)
//         getProcAddress("glDeleteProgram"))&&
//        (glDeleteShader = (PFNGLDELETESHADERPROC)
//         getProcAddress("glDeleteShader"))&&
//        (glDetachShader = (PFNGLDETACHSHADERPROC)
//         getProcAddress("glDetachShader"))&&
       (glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)
        getProcAddress("glDisableVertexAttribArray"))&&
       (glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)
        getProcAddress("glEnableVertexAttribArray"))&&
//        (glGetActiveAttrib = (PFNGLGETACTIVEATTRIBPROC)
//         getProcAddress("glGetActiveAttrib"))&&
//        (glGetActiveUinform = (PFNGLGETACTIVEUNIFORMPROC)
//         getProcAddress("glGetActiveUinform"))&&
//        (glGetAttachedShaders = (PFNGLGETATTACHEDSHADERSPROC)
//         getProcAddress("glGetAttachedShaders"))&&
//        (glGetAttribLocation = (PFNGLGETATTRIBLOCATIONPROC)
//         getProcAddress("glGetAttribLocation"))&&
       (glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)
        getProcAddress("glGetProgramInfoLog"))&&
       (glGetProgramiv = (PFNGLGETPROGRAMIVPROC)
        getProcAddress("glGetProgramiv"))&&
       (glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)
        getProcAddress("glGetShaderInfoLog"))&&
       (glGetShaderiv = (PFNGLGETSHADERIVPROC)
        getProcAddress("glGetShaderiv"))&&
//        (glGetShaderSource = (PFNGLGETSHADERSOURCEPROC)
//         getProcAddress("glGetShaderSource"))&&
//        (glGetUniformfv = (PFNGLGETUNIFORMFVPROC)
//         getProcAddress("glGetUniformfv"))&&
//        (glGetUniformiv = (PFNGLGETUNIFORMIVPROC)
//         getProcAddress("glGetUniformiv"))&&
       (glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)
        getProcAddress("glGetUniformLocation"))&&
//        (glGetVertexAttribdv = (PFNGLGETVERTEXATTRIBDVPROC)
//         getProcAddress("glGetVertexAttribdv"))&&
//        (glGetVertexAttribfv = (PFNGLGETVERTEXATTRIBFVPROC)
//         getProcAddress("glGetVertexAttribfv"))&&
//        (glGetVertexAttribiv = (PFNGLGETVERTEXATTRIBIVPROC)
//         getProcAddress("glGetVertexAttribiv"))&&
//        (glGetVertexAttribPointerv = (PFNGLGETVERTEXATTRIBPOINTERVPROC)
//         getProcAddress("glGetVertexAttribPointerv"))&&
//        (glIsProgram = (PFNGLISPROGRAMPROC)
//         getProcAddress("glIsProgram"))&&
//        (glIsShader = (PFNGLISSHADERPROC)
//         getProcAddress("glIsShader"))&&
       (glLinkProgram = (PFNGLLINKPROGRAMPROC)
        getProcAddress("glLinkProgram"))&&
       (glShaderSource = (PFNGLSHADERSOURCEPROC)
        getProcAddress("glShaderSource"))&&
       (glUseProgram = (PFNGLUSEPROGRAMPROC)
        getProcAddress("glUseProgram"))&&
       (glUniform1f = (PFNGLUNIFORM1FPROC)
        getProcAddress("glUniform1f"))&&
//        (glUniform2f = (PFNGLUNIFORM2FPROC)
//         getProcAddress("glUniform2f"))&&
//        (glUniform3f = (PFNGLUNIFORM3FPROC)
//         getProcAddress("glUniform3f"))&&
       (glUniform4f = (PFNGLUNIFORM4FPROC)
        getProcAddress("glUniform4f"))&&
//        (glUniform1i = (PFNGLUNIFORM1IPROC)
//         getProcAddress("glUniform1i"))&&
//        (glUniform2i = (PFNGLUNIFORM2IPROC)
//         getProcAddress("glUniform2i"))&&
//        (glUniform3i = (PFNGLUNIFORM3IPROC)
//         getProcAddress("glUniform3i"))&&
//        (glUniform4i = (PFNGLUNIFORM4IPROC)
//         getProcAddress("glUniform4i"))&&
//        (glUniform1fv = (PFNGLUNIFORM1FVPROC)
//         getProcAddress("glUniform1fv"))&&
//        (glUniform2fv = (PFNGLUNIFORM2FVPROC)
//         getProcAddress("glUniform2fv"))&&
//        (glUniform3fv = (PFNGLUNIFORM3FVPROC)
//         getProcAddress("glUniform3fv"))&&
//        (glUniform4fv = (PFNGLUNIFORM4FVPROC)
//         getProcAddress("glUniform4fv"))&&
//        (glUniform1iv = (PFNGLUNIFORM1IVPROC)
//         getProcAddress("glUniform1iv"))&&
//        (glUniform2iv = (PFNGLUNIFORM2IVPROC)
//         getProcAddress("glUniform2iv"))&&
//        (glUniform3iv = (PFNGLUNIFORM3IVPROC)
//         getProcAddress("glUniform3iv"))&&
//        (glUniform4iv = (PFNGLUNIFORM4IVPROC)
//         getProcAddress("glUniform4iv"))&&
//        (glUniformMatrix2fv = (PFNGLUNIFORMMATRIX2FVPROC)
//         getProcAddress("glUniformMatrix2fv"))&&
//        (glUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVPROC)
//         getProcAddress("glUniformMatrix3fv"))&&
//        (glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)
//         getProcAddress("glUniformMatrix4fv"))&&
//        (glValidateProgram = (PFNGLVALIDATEPROGRAMPROC)
//         getProcAddress("glValidateProgram"))&&
//        (glVertexAttrib1d = (PFNGLVERTEXATTRIB1DPROC)
//         getProcAddress("glVertexAttrib1d"))&&
//        (glVertexAttrib1dv = (PFNGLVERTEXATTRIB1DVPROC)
//         getProcAddress("glVertexAttrib1dv"))&&
//        (glVertexAttrib1f = (PFNGLVERTEXATTRIB1FPROC)
//         getProcAddress("glVertexAttrib1f"))&&
//        (glVertexAttrib1fv = (PFNGLVERTEXATTRIB1FVPROC)
//         getProcAddress("glVertexAttrib1fv"))&&
//        (glVertexAttrib1s = (PFNGLVERTEXATTRIB1SPROC)
//         getProcAddress("glVertexAttrib1s"))&&
//        (glVertexAttrib1sv = (PFNGLVERTEXATTRIB1SVPROC)
//         getProcAddress("glVertexAttrib1sv"))&&
//        (glVertexAttrib2d = (PFNGLVERTEXATTRIB2DPROC)
//         getProcAddress("glVertexAttrib2d"))&&
//        (glVertexAttrib2dv = (PFNGLVERTEXATTRIB2DVPROC)
//         getProcAddress("glVertexAttrib2dv"))&&
//        (glVertexAttrib2f = (PFNGLVERTEXATTRIB2FPROC)
//         getProcAddress("glVertexAttrib2f"))&&
//        (glVertexAttrib2fv = (PFNGLVERTEXATTRIB2FVPROC)
//         getProcAddress("glVertexAttrib2fv"))&&
//        (glVertexAttrib2s = (PFNGLVERTEXATTRIB2SPROC)
//         getProcAddress("glVertexAttrib2s"))&&
//        (glVertexAttrib2sv = (PFNGLVERTEXATTRIB2SVPROC)
//         getProcAddress("glVertexAttrib2sv"))&&
//        (glVertexAttrib3d = (PFNGLVERTEXATTRIB3DPROC)
//         getProcAddress("glVertexAttrib3d"))&&
//        (glVertexAttrib3dv = (PFNGLVERTEXATTRIB3DVPROC)
//         getProcAddress("glVertexAttrib3dv"))&&
//        (glVertexAttrib3f = (PFNGLVERTEXATTRIB3FPROC)
//         getProcAddress("glVertexAttrib3f"))&&
//        (glVertexAttrib3fv = (PFNGLVERTEXATTRIB3FVPROC)
//         getProcAddress("glVertexAttrib3fv"))&&
//        (glVertexAttrib3s = (PFNGLVERTEXATTRIB3SPROC)
//         getProcAddress("glVertexAttrib3s"))&&
//        (glVertexAttrib3sv = (PFNGLVERTEXATTRIB3SVPROC)
//         getProcAddress("glVertexAttrib3sv"))&&
//        (glVertexAttrib4Nbv = (PFNGLVERTEXATTRIB4NBVPROC)
//         getProcAddress("glVertexAttrib4Nbv"))&&
//        (glVertexAttrib4Niv = (PFNGLVERTEXATTRIB4NIVPROC)
//         getProcAddress("glVertexAttrib4Niv"))&&
//        (glVertexAttrib4Nsv = (PFNGLVERTEXATTRIB4NSVPROC)
//         getProcAddress("glVertexAttrib4Nsv"))&&
//        (glVertexAttrib4Nub = (PFNGLVERTEXATTRIB4NUBPROC)
//         getProcAddress("glVertexAttrib4Nub"))&&
//        (glVertexAttrib4Nubv = (PFNGLVERTEXATTRIB4NUBVPROC)
//         getProcAddress("glVertexAttrib4Nubv"))&&
//        (glVertexAttrib4Nuiv = (PFNGLVERTEXATTRIB4NUIVPROC)
//         getProcAddress("glVertexAttrib4Nuiv"))&&
//        (glVertexAttrib4Nusv = (PFNGLVERTEXATTRIB4NUSVPROC)
//         getProcAddress("glVertexAttrib4Nusv"))&&
//        (glVertexAttrib4bv = (PFNGLVERTEXATTRIB4BVPROC)
//         getProcAddress("glVertexAttrib4bv"))&&
//        (glVertexAttrib4d = (PFNGLVERTEXATTRIB4DPROC)
//         getProcAddress("glVertexAttrib4d"))&&
//        (glVertexAttrib4dv = (PFNGLVERTEXATTRIB4DVPROC)
//         getProcAddress("glVertexAttrib4dv"))&&
//        (glVertexAttrib4f = (PFNGLVERTEXATTRIB4FPROC)
//         getProcAddress("glVertexAttrib4f"))&&
//        (glVertexAttrib4fv = (PFNGLVERTEXATTRIB4FVPROC)
//         getProcAddress("glVertexAttrib4fv"))&&
//        (glVertexAttrib4iv = (PFNGLVERTEXATTRIB4IVPROC)
//         getProcAddress("glVertexAttrib4iv"))&&
//        (glVertexAttrib4s = (PFNGLVERTEXATTRIB4SPROC)
//         getProcAddress("glVertexAttrib4s"))&&
//        (glVertexAttrib4sv = (PFNGLVERTEXATTRIB4SVPROC)
//         getProcAddress("glVertexAttrib4sv"))&&
//        (glVertexAttrib4ubv = (PFNGLVERTEXATTRIB4UBVPROC)
//         getProcAddress("glVertexAttrib4ubv"))&&
//        (glVertexAttrib4uiv = (PFNGLVERTEXATTRIB4UIVPROC)
//         getProcAddress("glVertexAttrib4uiv"))&&
//        (glVertexAttrib4usv = (PFNGLVERTEXATTRIB4USVPROC)
//         getProcAddress("glVertexAttrib4usv"))&&
       (glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERVPROC)
        getProcAddress("glVertexAttribPointer")));
  }
  return shader_functions_built_;
}

#endif




// ************* Persistant stuff needs implementation ************** //

#define PARTICLEFLOWRENDERER_VERSION 1
void 
ParticleFlowRenderer::io(Piostream&)
{
  // nothing for now...
  NOT_FINISHED("ParticleFlowRenderer::io");
}

bool
ParticleFlowRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
  NOT_FINISHED("ParticleFlowRenderer::saveobj");
  return false;
}
// ****************************************************************** //


GLint 
ParticleFlowShader::getUniLoc(GLuint program, const GLchar *name)
{
    GLint loc;

    loc = glGetUniformLocation(program, name);

    if (loc == -1)
        printf("No such uniform named \"%s\"\n", name);

    printOpenGLError();  // Check for OpenGL errors
    return loc;
}

void 
ParticleFlowShader::printShaderInfoLog(GLuint shader)
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = new GLchar[infologLength];
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        printf("Shader InfoLog:\n%s\n\n", infoLog);
        delete infoLog;
    }
    printOpenGLError();  // Check for OpenGL errors
}

void 
ParticleFlowShader::printProgramInfoLog(GLuint program)
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors

    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = new GLchar[infologLength];
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
        printf("Program InfoLog:\n%s\n\n", infoLog);
        delete infoLog;
    }
    printOpenGLError();  // Check for OpenGL errors
}

int 
ParticleFlowShader::shaderSize(char *fileName, EShaderType shaderType)
{
    //
    // Returns the size in bytes of the shader fileName.
    // If an error occurred, it returns -1.
    //
    // File name convention:
    //
    // <fileName>.vert
    // <fileName>.frag
    //
    int fd;
    char name[100];
    int count = -1;

    strcpy(name, fileName);

    switch (shaderType)
    {
        case EVertexShader:
            strcat(name, ".vert");
            break;
        case EFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            printf("ERROR: unknown shader file type\n");
            exit(1);
            break;
    }
    //
    // Open the file, seek to the end to find its length
    //
#ifdef WIN32 /*[*/
    fd = _open(name, _O_RDONLY);
    if (fd != -1)
    {
        count = _lseek(fd, 0, SEEK_END) + 1;
        _close(fd);
    }
#else /*][*/
    fd = open(name, O_RDONLY);
    if (fd != -1)
    {
        count = lseek(fd, 0, SEEK_END) + 1;
        close(fd);
    }
#endif /*]*/

    return count;
}

int 
ParticleFlowShader::readShader(char *fileName, EShaderType shaderType,
                               char *shaderText, int size)
{
    //
    // Reads a shader from the supplied file and returns the shader in the
    // arrays passed in. Returns 1 if successful, 0 if an error occurred.
    // The parameter size is an upper limit of the amount of bytes to read.
    // It is ok for it to be too big.
    //
    FILE *fh;
    char name[100];
    int count;

    strcpy(name, fileName);

    switch (shaderType)
    {
        case EVertexShader:
            strcat(name, ".vert");
            break;
        case EFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            printf("ERROR: unknown shader file type\n");
            exit(1);
            break;
    }

    //
    // Open the file
    //
    fh = fopen(name, "r");
    if (!fh)
        return -1;

    //
    // Get the shader from a file.
    //
    fseek(fh, 0, SEEK_SET);
    count = (int) fread(shaderText, 1, size, fh);
    shaderText[count] = '\0';

    if (ferror(fh))
        count = 0;

    fclose(fh);
    return count;
}

int
ParticleFlowShader::readShaderSource(char *fileName, GLchar **vertexShader,
                                     GLchar **fragmentShader)
{
    int vSize, fSize;

    //
    // Allocate memory to hold the source of our shaders.
    //
    vSize = shaderSize(fileName, EVertexShader);
    fSize = shaderSize(fileName, EFragmentShader);

    if ((vSize == -1) || (fSize == -1))
    {
        printf("Cannot determine size of the shader %s\n", fileName);
        return 0;
    }

    *vertexShader = (GLchar *) malloc(vSize);
    *fragmentShader = (GLchar *) malloc(fSize);

    //
    // Read the source code
    //
    if (!readShader(fileName, EVertexShader, *vertexShader, vSize))
    {
        printf("Cannot read the file %s.vert\n", fileName);
        return 0;
    }

    if (!readShader(fileName, EFragmentShader, *fragmentShader, fSize))
    {
        printf("Cannot read the file %s.frag\n", fileName);
        return 0;
    }

    return 1;
}

int 
ParticleFlowShader::installParticleShaders(const GLchar *particleVertex,
                                           const GLchar *particleFragment)
{

  GLint       vertCompiled, fragCompiled;    // status values
  GLint       linked;

  // Create a vertex shader object and a fragment shader object

  VertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
  FragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

  // Load source code strings into shaders

  glShaderSource(VertexShaderObject, 1, &particleVertex, NULL);
  glShaderSource(FragmentShaderObject, 1, &particleFragment, NULL);

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(VertexShaderObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(VertexShaderObject, GL_COMPILE_STATUS, &vertCompiled);
  printShaderInfoLog(VertexShaderObject);

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(FragmentShaderObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(FragmentShaderObject, GL_COMPILE_STATUS, &fragCompiled);
  printShaderInfoLog(FragmentShaderObject);

  if (!vertCompiled || !fragCompiled)
    return 0;

  // Create a program object and attach the two compiled shaders

  ProgramObject = glCreateProgram();
  glAttachShader(ProgramObject, VertexShaderObject);
  glAttachShader(ProgramObject, FragmentShaderObject);

  // Bind generic attribute indices to attribute variable names

  glBindAttribLocation(ProgramObject, VELOCITY_ARRAY, "Velocity");
  glBindAttribLocation(ProgramObject, START_TIME_ARRAY, "StartTime");

  // Link the program object and print out the info log

  glLinkProgram(ProgramObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetProgramiv(ProgramObject, GL_LINK_STATUS, &linked);
  printProgramInfoLog(ProgramObject);

  if (!linked)
    return 0;

  // Install program object as part of current state

  glUseProgram(ProgramObject);

  // Set up initial uniform values

  glUniform4f(getUniLoc(ProgramObject, "Background"), 0.0, 0.0, 0.0, 1.0);
  printOpenGLError();  // Check for OpenGL errors
  glUniform1f(getUniLoc(ProgramObject, "Time"), -5.0);
  printOpenGLError();  // Check for OpenGL errors

  glUseProgram(0);
  return 1;
}
