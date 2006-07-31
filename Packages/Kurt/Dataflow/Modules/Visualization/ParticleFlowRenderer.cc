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

#include <Packages/Kurt/Dataflow/Modules/Visualization/ParticleFlowRenderer.h>
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


#include <sys/types.h>
#include <unistd.h>

#if !defined(HAVE_GLEW)
#error GLEW is not defined
#endif

using namespace Kurt;
using namespace SCIRun;
using std::cerr;

extern int printOglError(char *file, int line);

#ifndef  printOpenGLError
#define printOpenGLError() printOglError(__FILE__, __LINE__)
#endif



// static members
bool ParticleFlowRenderer::functions_initialized_(false);

ParticleFlowRenderer::ParticleFlowRenderer() :
  GeomObj(),
  initialized_(false),
  vshader_initialized_(false),
  cmap_dirty_(true),
  flow_tex_dirty_(true),
  part_tex_dirty_(true),
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
  array_width_(16),
  array_height_(16),
  start_verts_(0),
  current_verts_(0),
  colors_(0),
  velocities_(0),
  start_times_(0),
  fb_(0)
{

  //Set up the initial particle transformation based on how we build
  // our particles in create_points()
  Point unused;
  initial_particle_trans_.load_identity();
  Transform s;
  s.load_frame(unused, Vector(0.5, 0, 0), 
               Vector(0, 0.5, 0), Vector(0, 0, 0.5));
  initial_particle_trans_.pre_trans( s );

}

void print_array( int size, int values,  GLfloat *array )
{
  GLfloat *ptr = array;
  for(int i = 0; i < size; i++){
    cerr<<"(";
    for(int j = 0; j < values; j++){
    cerr<<*ptr<<", ";
    ++ptr;
    }
    cerr<<") \n";
  }
  cerr<<"\n";
}


ParticleFlowRenderer::ParticleFlowRenderer(const ParticleFlowRenderer& copy):
  initialized_(copy.initialized_),
  vshader_initialized_(copy.vshader_initialized_),

  cmap_dirty_(copy.cmap_dirty_),
  flow_tex_dirty_(copy.flow_tex_dirty_),
  part_tex_dirty_(copy.part_tex_dirty_),
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
  start_verts_(copy.start_verts_),
  current_verts_(copy.current_verts_),
  colors_(copy.colors_),
  velocities_(copy.velocities_),
  start_times_(copy.start_times_),
  fb_(copy.fb_)
{}

ParticleFlowRenderer::~ParticleFlowRenderer()
{
  cmap_h_ = 0;
  fh_ = 0;
  delete shader_;
  delete vshader_;
  delete fb_;
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
}

void 
ParticleFlowRenderer::update_vector_field( FieldHandle vfh, bool normalize )
{
  fh_ = vfh;

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
    flow_tex_dirty_ = true;
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
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    draw_flow_outline();
  } else {
#if 1
    if(!initialized_){
      if( /* cmap_h_ == 0 || */ fh_ == 0){
        return;
      }

      cerr<<"pid is "<<getpid()<<"\n";

      // set up FBuffer as render to texture
      //create reference textures
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      load_part_texture(0);
      load_part_texture(1);      
 
      // make sure there is an frame buffer object
      if( fb_ == 0) {
        fb_ = scinew Fbuffer(int(array_width_), int(array_height_));
      }
      CHECK_OPENGL_ERROR("");
      
      if ( !Fbuffer_configure() ){
        cerr<<"invalid frame buffer object.  Do nothing.\n";
        return;
      }
      CHECK_OPENGL_ERROR("");
      if ( !shader_configure() ) {
        cerr<<"shader initialization failed.  Do nothing.\n";
        return;
      }
      CHECK_OPENGL_ERROR("");
      
      // create_points for rendering
      create_points(array_width_, array_height_);
      
      cerr<<"Starting array values are:  \n";
      print_array(array_width_*array_height_, 4, start_verts_);
      initialized_ = true;
    }
#endif

    

    if( part_tex_dirty_ ){
      load_part_texture( 0, start_verts_);
      load_part_texture( 1, current_verts_);
      part_tex_dirty_ = false;
    }
    else {
      load_part_texture( 1, current_verts_);
    }



    if( flow_tex_dirty_ ){
      reload_flow_texture();
    }


    CHECK_OPENGL_ERROR("");
    glMatrixMode(GL_MODELVIEW);
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // save the current draw buffer
    glGetIntegerv(GL_DRAW_BUFFER, &current_draw_buffer_);

    // render to the frame buffer object
    fb_->enable();
    // draw to texture 
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);  
    // get the current viewport dimensions
    // check the frame buffer again.
    fb_->check_buffer();

    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    // set the viewport to texture size
    glViewport(0,0, array_width_, array_height_);

    cerr<<"Current array values are:  \n";
    print_array(array_width_*array_height_, 4, current_verts_);

    CHECK_OPENGL_ERROR("");
    // bind the textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, part_tex_[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, part_tex_[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_3D, flow_tex_);
    CHECK_OPENGL_ERROR("");      
    glActiveTexture(GL_TEXTURE1);
    double m[16];
    glGetDoublev(GL_PROJECTION_MATRIX, m);
//     CHECK_OPENGL_ERROR("");      
    glMatrixMode(GL_PROJECTION);
//     CHECK_OPENGL_ERROR("");      
    glLoadIdentity();
//     CHECK_OPENGL_ERROR("");      
    gluOrtho2D(0,1,0,1);
//     CHECK_OPENGL_ERROR("");      
    glMatrixMode(GL_MODELVIEW);
//     CHECK_OPENGL_ERROR("");    
    glPushMatrix();
//     CHECK_OPENGL_ERROR("");    
    glLoadIdentity();
    CHECK_OPENGL_ERROR("");    

    {
      GLdouble mat[16];
      shader_trans_.get_trans( mat );
      Transform  tform;
      fh_->mesh()->get_canonical_transform(tform);
      double mvmat[16];
      tform.invert();
      tform.get_trans(mvmat);
      GLfloat matf[16];
      GLfloat mvmatf[16];
      cerr<<"Widget Transform:\n";
      for(int i = 0; i < 16; i++){
        matf[i] = (float) mat[i];
        cerr<<matf[i];
        if( (i+1) % 4 == 0 ){
          cerr<<"\n";
        } else {
          cerr<<", ";
        }
      }
      cerr<<"Canonical Mesh Transform:\n";
      for(int i = 0; i < 16; i++){
        mvmatf[i] = (float) mvmat[i];
        cerr<<mvmatf[i];
        if( (i+1) % 4 == 0 ){
          cerr<<"\n";
        } else {
          cerr<<", ";
        }
      }

      Transform mtform;
      double mmmat[16];
      fh_->mesh()->transform(mtform);
      mtform.post_trans( tform );
      mtform.get_trans(mmmat);
      GLfloat mmmatf[16];
      cerr<<"Mesh Transform:\n";
      for(int i = 0; i < 16; i++){
        mmmatf[i] = float(mmmat[i]);
        cerr<<mmmatf[i];
        if( (i+1) % 4 == 0 ){
          cerr<<"\n";
        } else {
          cerr<<", ";
        }
      }
      
      //   // Set up initial uniform values
      shader_->initialize_uniform( "ParticleTrans", 1, false, &(matf[0]));
      shader_->initialize_uniform( "MeshTrans", 1, false, &(mmmatf[0]));
//       shader_->initialize_uniform( "MeshSize", GLfloat(nx_), GLfloat(ny_),
//                                    GLfloat(nz_) );
      shader_->initialize_uniform( "Time", GLfloat(-0.05));
      shader_->initialize_uniform( "StartPositions", 0 );
      shader_->initialize_uniform( "Positions", 1 );      
      shader_->initialize_uniform( "Flow", 2 );      
    }
    printOpenGLError();

    // use shader_ to move points in framebuffer object
    glUseProgram(shader_->program_object());

    glBegin(GL_QUADS);
    {
      glMultiTexCoord2f(GL_TEXTURE0, 0, 0); glVertex2f(0, 0);
      glMultiTexCoord2f(GL_TEXTURE0, 1, 0); glVertex2f(1, 0);
      glMultiTexCoord2f(GL_TEXTURE0, 1, 1); glVertex2f(1, 1);
      glMultiTexCoord2f(GL_TEXTURE0, 0, 1); glVertex2f(0, 1);
    }
    glEnd();
    printOpenGLError();
    glPopMatrix(); //MODELVIEW
    CHECK_OPENGL_ERROR("");    
    glMatrixMode(GL_PROJECTION);
    CHECK_OPENGL_ERROR("");    
    glLoadMatrixd( m );
    glMatrixMode(GL_MODELVIEW);
    CHECK_OPENGL_ERROR("");    
    // disable the shader
    glUseProgram(0);
    CHECK_OPENGL_ERROR("");    
    // undo the texture bindings
    glActiveTexture(GL_TEXTURE0_ARB);
//     glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1_ARB);
//     glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2_ARB);
//     glDisable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, 0);
    glActiveTexture(GL_TEXTURE0);
    CHECK_OPENGL_ERROR(""); 

    // read the pixels from the framebuffer object
    glReadPixels(0,0, array_width_, array_height_,
                 GL_RGBA, GL_FLOAT, current_verts_);
    cerr<<"Current array values are now: \n";
    print_array(array_width_*array_height_, 4, current_verts_);

    // disable the framebuffer object
    fb_->disable();
    // reset the draw buffer parameters
    glDrawBuffer( current_draw_buffer_);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    CHECK_OPENGL_ERROR("");    
    // Prepare to render points.
    glEnable(GL_DEPTH_TEST);
    GLboolean lighting = glIsEnabled(GL_LIGHTING);
    glDisable(GL_LIGHTING);
    glColor4f(1.0,1.0,1.0,1.0);
    CHECK_OPENGL_ERROR("");
    if(lighting) glEnable(GL_LIGHTING);

    draw_points();
    CHECK_OPENGL_ERROR("");
  }
  di = 0;
}

bool
ParticleFlowRenderer::Fbuffer_configure() {

      fb_->create();
      fb_->enable();  //Bind framebuffer object.
      
      // Attach texture to framebuffer color buffer, draw to the positions
      // texture.
      fb_->attach_texture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, part_tex_[1]);
      fb_->disable();
      
      if( !fb_->check_buffer() ) { 
        cerr<<"invalid frame buffer object.  Do nothing.\n";
        return false;
      }

      return true;
}


bool
ParticleFlowRenderer::shader_configure()
{
  GLchar *vs, *fs;
  shader_ = scinew GLSLShader();
  shader_->read_shader_source("advect", &vs, &fs);
  shader_->install_shaders(vs, fs);
  int success = shader_->compile_shaders();
      
  if( success ) {
    shader_->create_program();
    shader_->attach_objects();
//     shader_->initialize_uniform( "texUnit", 1 );
//     shader_->initialize_uniform( "vecUnit", 2 );
    success = shader_->link_program();
  }

  if( !success ){
    cerr<<"Shader installation failed\n";
    delete shader_;
    shader_ = 0;
    return false;
  }

  return true;
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

void ParticleFlowRenderer::load_part_texture(GLuint unit, float *verts)
{
  if(glIsTexture(part_tex_[unit])){
    glDeleteTextures(1, &(part_tex_[unit]));
  }
  glActiveTexture(GL_TEXTURE0_ARB+unit);
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &part_tex_[unit]);
  glBindTexture(GL_TEXTURE_2D, part_tex_[unit]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, array_width_, array_height_,
               0, GL_RGBA, GL_FLOAT, verts); 

  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  glActiveTexture(GL_TEXTURE0_ARB);
    

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
    
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F_ARB, nx_, ny_, nz_, 0,
                 GL_RGB, GL_FLOAT, vfield_); 

    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE0_ARB);
    
    flow_tex_dirty_ = false;
  }
}


// void  
// ParticleFlowRenderer::createPoints(GLint w, GLint h)
// {
// 	GLfloat *vptr, *cptr, *cvptr, *velptr, *stptr;
// 	GLfloat i, j;

// 	if (start_verts_ != 0) 
//           delete start_verts_;
//         if (colors_ != 0)
//           delete colors_;
//         if (velocities_ != 0)
//           delete velocities_;
//         if (start_times_ != 0)
//           delete start_times_;

// 	start_verts_  = scinew GLfloat[w * h * 3 * sizeof(float)];
//         current_verts_  = scinew GLfloat[w * h * 3 * sizeof(float)];
// 	colors_ = scinew GLfloat[w * h * 3 * sizeof(float)];
// 	velocities_ = scinew GLfloat[w * h * 3 * sizeof(float)];
// 	start_times_ = scinew GLfloat[w * h * sizeof(float)];

// 	vptr = start_verts_;
//         cvptr = current_verts_;
// 	cptr = colors_;
// 	velptr = velocities_;
// 	stptr  = start_times_;

// 	for (i = 0.5 / w - 0.5; i < 0.5; i = i + 1.0/w)
// 		for (j = 0.5 / h - 0.5; j < 0.5; j = j + 1.0/h)
// 		{
// 			*vptr       = i;
// 			*(vptr + 1) = 0.0;
// 			*(vptr + 2) = j;
// 			vptr += 3;

//                         *cvptr = 0.0; *(cvptr+1) = 0.0; 
//                         *(cvptr+2) = 0.0; cvptr += 3;

// 			*cptr       = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
// 			*(cptr + 1) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
// 			*(cptr + 2) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
// 			cptr += 3;

// 			*velptr       = (((float) rand() / RAND_MAX)) + 3.0;
// 			*(velptr + 1) =  ((float) rand() / RAND_MAX) * 10.0;
// 			*(velptr + 2) = (((float) rand() / RAND_MAX)) + 3.0;
// 			velptr += 3;

// 			*stptr = ((float) rand() / RAND_MAX) * 10.0;
// 			stptr++;
// 		}

// 	array_width_  = w;
// 	array_height_ = h;
// }


void  
ParticleFlowRenderer::create_points(GLint w, GLint h)
{
  GLfloat *vptr, *cptr, *cvptr;
  GLfloat i, j;

  if (start_verts_ != 0) 
    delete start_verts_;
  if (current_verts_ != 0)
    delete current_verts_;
  if ( colors_ != 0 )
    delete colors_;

  start_verts_  = scinew GLfloat[w * h * 4];
  current_verts_  = scinew GLfloat[w * h * 4];
  colors_ = scinew GLfloat[w * h * 3];

  vptr = start_verts_;
  cvptr = current_verts_;
  cptr = colors_;

  for (i = 0.5/w - 0.5; i < 0.5; i += 1.0/w ){
    for (j = 0.5/h - 0.5; j < 0.5; j += 1.0/h )
     {
      *vptr       = i;
      *(vptr + 1) = 0;
      *(vptr + 2) = j;
      *(vptr + 3) = 0.75 + ((float) rand() / RAND_MAX) * 0.5;
      vptr += 4;

      *cvptr       = 0.0;
      *(cvptr + 1) = 0.0;
      *(cvptr + 2) = 0.0;
      *(cvptr + 3) = 0.0;      
      cvptr += 4;

      *cptr       = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
      *(cptr + 1) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
      *(cptr + 2) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
      cptr += 3;
    }
  }
  
}


void 
ParticleFlowRenderer::updateAnim()
{
CHECK_OPENGL_ERROR("");

  particle_time_ += time_increment_;
  if (particle_time_ > 15.0)
    particle_time_ = 0.0;
  
  shader_->initialize_uniform("Time", particle_time_);
  
  CHECK_OPENGL_ERROR("");
}

bool
ParticleFlowRenderer::vshader_configure()
{
  GLchar *vs, *fs;
  vshader_ = scinew GLSLShader();
  vshader_->read_shader_source("particle", &vs, &fs);
  vshader_->install_shaders(vs, fs);
  int success = vshader_->compile_shaders();
      
  if( success ) {
    vshader_->create_program();
    vshader_->attach_objects();
//     vshader_->bind_attribute_location( START_POSITION_ARRAY, "StartPosition" );
    success = vshader_->link_program();
  }
  
  if( success ) {
    //   // Set up initial uniform values
    vshader_->initialize_uniform( "Background", 0.0, 0.0, 0.0, 0.0);
  }
  if( !success ){
    cerr<<"Shader installation failed\n";
    delete vshader_;
    vshader_ = 0;
    return false;
  }

  return true;
}



// void  
// ParticleFlowRenderer::drawPoints()
// {	
  
//   if( !have_shader_functions() ){
//       cerr<<"shader functions not available\n";
//       return; // do nothing
//     }

//   GLchar *vertex_shader_source, *fragment_shader_source;

//   if( shader_ == 0 ){
//     shader_ = scinew GLSLShader();
//     shader_->read_shader_source("particle",&vertex_shader_source, 
//                                          &fragment_shader_source);
//     shader_->install_shaders(vertex_shader_source,
//                              fragment_shader_source);

//     int success = shader_->compile_shaders();
    
//     if( success ) {
//       shader_->create_program();
//       shader_->attach_objects();
//       shader_->bind_attribute_location ( VELOCITY_ARRAY, "Velocity" );
//       shader_->bind_attribute_location ( START_TIME_ARRAY, "StartTime" );
//       success = shader_->link_program();
//     }
//     if( success ) {
//       //   // Set up initial uniform values
//       shader_->initialize_uniform( "Background", 0.0, 0.0, 0.0, 0.0);
//       shader_->initialize_uniform( "Time", GLfloat(-5.0));
//     }
//     if( !success ){
//       cerr<<"Shader installation failed\n";
//       delete shader_;
//       shader_ = 0;
//       return;
//     }
//   }
//    glMatrixMode(GL_MODELVIEW);
//    glPushMatrix();
//    //   glLoadIdentity();
   
//    // set up transform for shader
//    GLdouble mat[16];
//    shader_trans_.get_trans( mat );
// //    for(int i = 0; i < 4; i++) {
// //      for( int j = 0; j< 4; j++) {
// //        cerr<< mat[i*4 + j]<< " ";
// //      }
// //      cerr<<"\n";
// //    }
// //    cerr<<"\n";
//    glMultMatrixd( mat );

//    glUseProgram(shader_->ProgramObject);

//   if(animating_) updateAnim();

  


//   GLboolean depth;
//   glGetBooleanv(GL_DEPTH_TEST, &depth);
//   glDepthFunc(GL_LESS);

//   if( !depth ){
//     glEnable(GL_DEPTH_TEST);
//   }


  
//   glPointSize(2.0);

//   glVertexPointer(3, GL_FLOAT, 0, start_verts_);
//   glColorPointer(3, GL_FLOAT, 0, colors_);
// //    glVertexAttribPointer(VELOCITY_ARRAY,  3, GL_FLOAT, GL_FALSE, 0, velocities_);
// //   glVertexAttribPointer(START_TIME_ARRAY, 1, GL_FLOAT, GL_FALSE, 0, start_times_);

//   glEnableClientState(GL_VERTEX_ARRAY);
//   glEnableClientState(GL_COLOR_ARRAY);
//   glEnableVertexAttribArray(VELOCITY_ARRAY);
//   glEnableVertexAttribArray(START_TIME_ARRAY);

//   glDrawArrays(GL_POINTS, 0, array_width_ * array_height_);

//   glDisableClientState(GL_VERTEX_ARRAY);
//   glDisableClientState(GL_COLOR_ARRAY);
//   glDisableVertexAttribArray(VELOCITY_ARRAY);
//   glDisableVertexAttribArray(START_TIME_ARRAY);
  
//   if( !depth ){
//     glDisable(GL_DEPTH_TEST);
//   }
  
//   glUseProgram(0);
//   glPopMatrix();
// }



void  
ParticleFlowRenderer::draw_points()
{	

  if( !have_shader_functions() ){
      cerr<<"shader functions not initialized\n";
      return; // do nothing
  }
 
  if( !vshader_initialized_ ) {
    if( !vshader_configure() ){
      cerr<<"vertex shader didn't configure properly\n";
      return;
    } else {
      vshader_initialized_ = true;
    }
  }
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   //   glLoadIdentity();
   
   // set up transform for shader
   GLdouble mat[16];
   shader_trans_.get_trans( mat );
//    for(int i = 0; i < 4; i++) {
//      for( int j = 0; j< 4; j++) {
//        cerr<< mat[i*4 + j]<< " ";
//      }
//      cerr<<"\n";
//    }
//    cerr<<"\n";
//    glMultMatrixd( mat );

   glUseProgram(vshader_->program_object());
CHECK_OPENGL_ERROR("");

  if(animating_) updateAnim();
CHECK_OPENGL_ERROR("");

  

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

  glVertexPointer(3, GL_FLOAT, 4* sizeof(GLfloat), current_verts_);
CHECK_OPENGL_ERROR("");

  glColorPointer(3, GL_FLOAT, 0, colors_);
CHECK_OPENGL_ERROR("");
//   glVertexAttribPointer(VELOCITY_ARRAY,  3, GL_FLOAT, GL_FALSE, 0, velocities_);
//   glVertexAttribPointer(START_TIME_ARRAY, 1, GL_FLOAT, GL_FALSE, 0, start_times_);
//   glVertexAttribPointer(START_POSITION_ARRAY, 1, GL_FLOAT, GL_FALSE, 
//                         0, start_verts_);


  glEnableClientState(GL_VERTEX_ARRAY);
CHECK_OPENGL_ERROR("");
  glEnableClientState(GL_COLOR_ARRAY);
CHECK_OPENGL_ERROR("");

//   glEnableVertexAttribArray(VELOCITY_ARRAY);
//   glEnableVertexAttribArray(START_TIME_ARRAY);
//   glEnableVertexAttribArray(START_POSITION_ARRAY);

  glDrawArrays(GL_POINTS, 0, array_width_ * array_height_);
CHECK_OPENGL_ERROR("");

  glDisableClientState(GL_VERTEX_ARRAY);
CHECK_OPENGL_ERROR("");
  glDisableClientState(GL_COLOR_ARRAY);
CHECK_OPENGL_ERROR("");
//   glDisableVertexAttribArray(VELOCITY_ARRAY);
//   glDisableVertexAttribArray(START_TIME_ARRAY);
//   glDisableVertexAttribArray(START_POSITION_ARRAY);
  
  if( !depth ){
    glDisable(GL_DEPTH_TEST);
  }
  
  glUseProgram(0);
CHECK_OPENGL_ERROR("");
  glPopMatrix();
}


void
ParticleFlowRenderer::update_transform(const Point& c, const Point& r, 
                                       const Point& d)
{
  cerr<<"Updating widget transform.\n";
  Transform s;
  Point unused;
  shader_trans_.load_identity();
  shader_trans_.pre_scale( Vector( (r - c).length(),
                                   (Cross (d-c, r-c)).length(),
                                   (d - c).length()));

  s.load_frame(unused, (r-c).normal(),
               (Cross (d-c, r-c)).normal(), (d-c).normal() ); 
  shader_trans_.pre_trans(s);
  shader_trans_.pre_translate( c.asVector() );

  Transform inv( initial_particle_trans_);
  inv.invert();
  shader_trans_.post_trans(inv);
}

bool
ParticleFlowRenderer::have_shader_functions()
{
#if !defined(HAVE_GLEW)
  // we are not going to support GLSL without GLEW at this point
  return false;
#else
  if( glewIsSupported("GL_VERSION_2_0 GL_EXT_framebuffer_object")) {
    // great OpenGL 2.0 and FrameBuffeObjects are supported
    return true;
  } else {
    return false;
  }
#endif

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

