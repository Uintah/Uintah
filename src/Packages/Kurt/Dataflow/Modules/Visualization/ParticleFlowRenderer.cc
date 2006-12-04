  
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

#define PB 1

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
  recompute_(true),
  frozen_(false),
  buffer_(LEFT),
  time_(0.0),
  time_increment_(-0.05f),
  nsteps_(1),
  step_size_(1.0),
  flow_tex_(0),
  vfield_(0),
  vb_(0),
  cmap_h_(0),
  fh_(0),
  di_(0),
  shader_(0),
  particle_power_(4),
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
  recompute_(copy.recompute_),
  frozen_(copy.frozen_),
  time_( copy.time_),
  time_increment_(copy.time_increment_),
  nsteps_(copy.nsteps_),
  step_size_(copy.step_size_),
  flow_tex_(copy.flow_tex_),
  vfield_(copy.vfield_),
  vb_(0),
  cmap_h_( copy.cmap_h_),
  fh_(copy.fh_),
  di_(copy.di_),
  shader_(copy.shader_),
  particle_power_(copy.particle_power_),
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

void 
ParticleFlowRenderer::set_particle_exponent( int e )
{ 
  if( particle_power_ != e) {
    recompute_ = true;
    array_width_ = int( pow(2.0, e) );
    array_height_ = int( pow(2.0, e) );
    particle_power_ = e;
  }
}
 

#define BUFFER_OFFSET(i)((char *)NULL + (i))
void 
ParticleFlowRenderer::draw(DrawInfoOpenGL* di, Material* mat, double /* time */)
{
  if(!pre_draw(di, mat, 0)) return;
  di_ = di;
  if(di->get_drawtype() == DrawInfoOpenGL::WireFrame ) {
    draw_flow_outline();
  } else {
#if 1
//     cerr<< "init = "<< initialized_<<", reset = "<<reset_<<
//       ",  recompute = "<<recompute_<<"\n";

    if(!initialized_ || reset_ || recompute_){
      if( /* cmap_h_ == 0 || */ fh_ == 0){
        return;
      }

      cerr<<"pid is "<<getpid()<<"\n";

      // set up FBuffer as render to texture
      //create reference textures
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      CHECK_OPENGL_ERROR("");
      load_part_texture(0);
      CHECK_OPENGL_ERROR("");
      load_part_texture(1);      
      CHECK_OPENGL_ERROR("");
 
      // make sure there is an frame buffer object
      if( fb_ == 0) {
        fb_ = scinew Fbuffer(int(array_width_) * 2, int(array_height_));
      } else {
        delete fb_;
        fb_ = scinew Fbuffer(int(array_width_) * 2, int(array_height_));
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
      CHECK_OPENGL_ERROR("");

      
      //       cerr<<"Starting array values are:  \n";
      //       print_array(array_width_*array_height_, 4, start_verts_);

#if PB
      if(glIsBuffer(vb_)){
	glDeleteBuffers(1, &vb_);
      }
      cerr<<"trying to use pixel buffer\n";
      glGenBuffers(1, &vb_);
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, vb_);
      glBufferData(GL_PIXEL_PACK_BUFFER_ARB, 
		   sizeof(GLfloat)* array_width_ * array_height_ *4,
		   NULL, GL_DYNAMIC_DRAW);
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
#endif
      initialized_ = true;
      reset_ = false;
    }
#endif



    if( !frozen_){
      if( part_tex_dirty_ ){
        cerr<<"particle texture is dirty\n";
        load_part_texture( 0, start_verts_);
#if !PB
 	load_part_texture( 1, current_verts_);
#endif
        part_tex_dirty_ = false;
	CHECK_OPENGL_ERROR("");
      }
      else {
#if PB
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, vb_);
#else
        load_part_texture( 1, current_verts_);
#endif
      CHECK_OPENGL_ERROR("");
      }



      if( flow_tex_dirty_ ){
        cerr<<"flow texture is dirty\n";
        reload_flow_texture();
      CHECK_OPENGL_ERROR("");
      }
      
      
      CHECK_OPENGL_ERROR("");
      glMatrixMode(GL_MODELVIEW);
      //    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
      
      // save the current draw buffer
      glGetIntegerv(GL_DRAW_BUFFER, &current_draw_buffer_);

      GLint current_read_buffer;
      glGetIntegerv(GL_READ_BUFFER, &current_read_buffer);

      // render to the frame buffer object
      fb_->enable();
      fb_->attach_texture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, part_tex_[1]);
      // draw to texture 
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);  

      // get the current viewport dimensions
      int vp[4];
      glGetIntegerv(GL_VIEWPORT, vp);
      // set the viewport to texture size
      if (buffer_ == LEFT)
	glViewport(0,0, array_width_, array_height_);
      else
	glViewport(array_width_,0, array_width_, array_height_);
      
      //     cerr<<"Current array values are:  \n";
      //     print_array(array_width_*array_height_, 4, current_verts_);


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
          CHECK_OPENGL_ERROR("");      
      glMatrixMode(GL_PROJECTION);
          CHECK_OPENGL_ERROR("");      
      glLoadIdentity();
          CHECK_OPENGL_ERROR("");      
      gluOrtho2D(0,1,0,1);
          CHECK_OPENGL_ERROR("");      
      glMatrixMode(GL_MODELVIEW);
          CHECK_OPENGL_ERROR("");    
      glPushMatrix();
          CHECK_OPENGL_ERROR("");    
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
        //       cerr<<"Widget Transform:\n";
//         for(int i = 0; i < 16; i++){
//           matf[i] = (float) mat[i];
//           //         cerr<<matf[i];
//           //         if( (i+1) % 4 == 0 ){
//           //           cerr<<"\n";
//           //         } else {
//           //           cerr<<", ";
//           //         }
//         }
//         //       cerr<<"Canonical Mesh Transform:\n";
//         for(int i = 0; i < 16; i++){
//           mvmatf[i] = (float) mvmat[i];
//           //         cerr<<mvmatf[i];
//           //         if( (i+1) % 4 == 0 ){
//           //           cerr<<"\n";
//           //         } else {
//           //           cerr<<", ";
//           //         }
//         }

        Transform mtform;
        double mmmat[16];
        fh_->mesh()->transform(mtform);
        mtform.post_trans( tform );
        mtform.get_trans(mmmat);
        GLfloat mmmatf[16];
        //       cerr<<"Mesh Transform:\n";
        for(int i = 0; i < 16; i++){
          matf[i] = (float) mat[i];
          mvmatf[i] = (float) mvmat[i];
          mmmatf[i] = float(mmmat[i]);
          //         cerr<<mmmatf[i];
          //         if( (i+1) % 4 == 0 ){
          //           cerr<<"\n";
          //         } else {
          //           cerr<<", ";
          //         }
        }

        //   // Set up initial uniform values
        shader_->initialize_uniform( "ParticleTrans", 1, false, &(matf[0]));
        shader_->initialize_uniform( "MeshTrans", 1, false, &(mmmatf[0]));
        shader_->initialize_uniform( "StartPositions", GLint(0) );
        shader_->initialize_uniform( "Positions", GLint(1) );      
        shader_->initialize_uniform( "Flow", GLint(2) );      
        shader_->initialize_uniform( "Time", GLfloat(time_increment_));
        shader_->initialize_uniform( "Step", GLfloat( step_size_));

      }

      CHECK_OPENGL_ERROR("");    

      // use shader_ to move points in framebuffer object
      glUseProgram(shader_->program_object());
      CHECK_OPENGL_ERROR("");    

      glBegin(GL_QUADS);
      {
	if( buffer_ == LEFT ) {
	  glMultiTexCoord2f(GL_TEXTURE0, 0, 0); 
	  glMultiTexCoord2f(GL_TEXTURE1, 0.5, 0); glVertex2f(0, 0);
	  glMultiTexCoord2f(GL_TEXTURE0, 1, 0);
	  glMultiTexCoord2f(GL_TEXTURE1, 1, 0);   glVertex2f(1, 0);
	  glMultiTexCoord2f(GL_TEXTURE0, 1, 1);
	  glMultiTexCoord2f(GL_TEXTURE1, 1, 1);   glVertex2f(1, 1);
	  glMultiTexCoord2f(GL_TEXTURE0, 0, 1); 
	  glMultiTexCoord2f(GL_TEXTURE1, 0.5, 1); glVertex2f(0, 1);
	} else {
	  glMultiTexCoord2f(GL_TEXTURE0, 0, 0); 
	  glMultiTexCoord2f(GL_TEXTURE1, 0, 0);	  glVertex2f(0, 0);
	  glMultiTexCoord2f(GL_TEXTURE0, 1, 0);
	  glMultiTexCoord2f(GL_TEXTURE1, 0.5, 0); glVertex2f(1, 0);
	  glMultiTexCoord2f(GL_TEXTURE0, 1, 1);
	  glMultiTexCoord2f(GL_TEXTURE1, 0.5, 1); glVertex2f(1, 1);
	  glMultiTexCoord2f(GL_TEXTURE0, 0, 1); 
	  glMultiTexCoord2f(GL_TEXTURE1, 0, 1);   glVertex2f(0, 1);
	}
      }
      glEnd();
      
      CHECK_OPENGL_ERROR("");    
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

      glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
      CHECK_OPENGL_ERROR("");    
      // read the pixels from the framebuffer object

#if PB
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, vb_);

      if( buffer_ == LEFT ) {
 	glReadPixels(0,0, array_width_, array_height_,
 		     GL_RGBA, GL_FLOAT, NULL);
      } else {//right
 	glReadPixels(array_width_,0, array_width_, array_height_,
 		     GL_RGBA, GL_FLOAT, NULL);
      }
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

#else
      // read the pixels from the framebuffer object
      if( buffer_ == LEFT ) {
	glReadPixels(0,0, array_width_, array_height_,
		     GL_RGBA, GL_FLOAT,  current_verts_);
      } else {//right
	glReadPixels(array_width_,0, array_width_, array_height_,
		     GL_RGBA, GL_FLOAT,  current_verts_);
      }
#endif
      CHECK_OPENGL_ERROR("");    

//           cerr<<"Current array values are now: \n";
//           print_array(array_width_*array_height_, 4, current_verts_);

      // disable the framebuffer object

//       fb_->unattach(GL_COLOR_ATTACHMENT0_EXT);
      fb_->disable();
      CHECK_OPENGL_ERROR("");    
      // reset the draw buffer parameters
      glDrawBuffer( current_draw_buffer_);
      glReadBuffer( current_read_buffer);
      glViewport(vp[0], vp[1], vp[2], vp[3]);
      CHECK_OPENGL_ERROR("");    
      // Prepare to render points.
      glEnable(GL_DEPTH_TEST);
      GLboolean lighting = glIsEnabled(GL_LIGHTING);
      glDisable(GL_LIGHTING);
      glColor4f(1.0,1.0,1.0,1.0);
      CHECK_OPENGL_ERROR("");
      if(lighting) glEnable(GL_LIGHTING);
      recompute_ = false;
    }
    draw_points();
    CHECK_OPENGL_ERROR("");

    if( buffer_ == LEFT ) buffer_ = RIGHT;
    else buffer_ = LEFT;
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
      } else {
        // fill the texture with zeros
        fb_->enable();
        // draw to texture 
        // get the current viewport dimensions
        int vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_DRAW_BUFFER, &current_draw_buffer_);
        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);  
        // set the viewport to texture size
        glViewport(0,0, 2*array_width_, array_height_);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, part_tex_[1]);
        double m[16];
        glGetDoublev(GL_PROJECTION_MATRIX, m);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        CHECK_OPENGL_ERROR("");      
        gluOrtho2D(0,1,0,1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor4f(0,0,0,0);
        glBegin(GL_QUADS);
	  glMultiTexCoord2f(GL_TEXTURE1, 0, 0); glVertex2f(0, 0);
	  glMultiTexCoord2f(GL_TEXTURE1, 1, 0); glVertex2f(1, 0);
	  glMultiTexCoord2f(GL_TEXTURE1, 1, 1); glVertex2f(1, 1);
	  glMultiTexCoord2f(GL_TEXTURE1, 0, 1); glVertex2f(0, 1);
        glEnd();
        glPopMatrix(); //MODELVIEW
        glBindTexture(GL_TEXTURE_2D,0);
        glActiveTexture(GL_TEXTURE0);
        glDrawBuffer( current_draw_buffer_);
        fb_->disable();
        glViewport(vp[0], vp[1], vp[2], vp[3]);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd( m );
        glMatrixMode(GL_MODELVIEW);
        CHECK_OPENGL_ERROR("");
        return true;
      }
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
    
  if( unit == 1 ) // hack
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, 2*array_width_, array_height_,
		 0, GL_RGBA, GL_FLOAT, verts); 
  else 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, array_width_, array_height_,
		 0, GL_RGBA, GL_FLOAT, verts); 
  CHECK_OPENGL_ERROR("");

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
    
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F_ARB, nx_, ny_, nz_, 0,
                 GL_RGB, GL_FLOAT, vfield_); 

    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    glActiveTexture(GL_TEXTURE0_ARB);
    
    flow_tex_dirty_ = false;
  }
}




void  
ParticleFlowRenderer::create_points(GLint w, GLint h)
{
  GLfloat *vptr, *cptr, *cvptr;
  GLfloat i, j;

  if (start_verts_ != 0) delete [] start_verts_;
  start_verts_  = scinew GLfloat[w * h * 4];
  vptr = start_verts_;
#if !PB
  if (current_verts_ != 0) delete [] current_verts_;
  current_verts_  = scinew GLfloat[w * h * 4];
  cvptr = current_verts_;
#endif
  if ( colors_ != 0 ) delete [] colors_;
  colors_ = scinew GLfloat[w * h * 3];
  cptr = colors_;


  for (i = 0.5/w - 0.5; i < 0.5; i += 1.0/w ){
    for (j = 0.5/h - 0.5; j < 0.5; j += 1.0/h )
     {
      *vptr       = i;
      *(vptr + 1) = 0;
      *(vptr + 2) = j;
      *(vptr + 3) = 0.75 + ((float) rand() / RAND_MAX) * 0.5;
      vptr += 4;

#if !PB
      *cvptr       = 0.0;
      *(cvptr + 1) = 0.0;
      *(cvptr + 2) = 0.0;
      *(cvptr + 3) = 0.0;      
      cvptr += 4;
#endif
      *cptr       = ((float) rand() / RAND_MAX) * 0.5 + 0.25;
      *(cptr + 1) = ((float) rand() / RAND_MAX) * 0.5 + 0.25;
      *(cptr + 2) = ((float) rand() / RAND_MAX) * 0.5 + 0.25;
      cptr += 3;
    }
  }
  part_tex_dirty_ = true;
}


void 
ParticleFlowRenderer::updateAnim()
{
//   cerr<<"shader 'Time' = "<<particle_time_<<" ";
  shader_->initialize_uniform("Time", time_increment_);
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

   glUseProgram(vshader_->program_object());
CHECK_OPENGL_ERROR("");

  if(animating_) updateAnim();
CHECK_OPENGL_ERROR("");

  

  GLboolean depth;
  glGetBooleanv(GL_DEPTH_TEST, &depth);
  glDepthFunc(GL_LESS);

  if( !depth ){
    glEnable(GL_DEPTH_TEST);
  }


  
  glPointSize(2.0);
  glColorPointer(3, GL_FLOAT, 0, colors_);
CHECK_OPENGL_ERROR("");

#if PB
  // reading from pixel buffer
  glBindBuffer(GL_ARRAY_BUFFER, vb_);
  glVertexPointer(3, GL_FLOAT, 4* sizeof(GLfloat), NULL);
// CHECK_OPENGL_ERROR("");
#else
  // reading from current_verts
  glVertexPointer(3, GL_FLOAT, 4* sizeof(GLfloat), current_verts_ );
#endif

CHECK_OPENGL_ERROR("");


  glEnableClientState(GL_VERTEX_ARRAY);
CHECK_OPENGL_ERROR("");
  glEnableClientState(GL_COLOR_ARRAY);
CHECK_OPENGL_ERROR("");

  glDrawArrays(GL_POINTS, 0, array_width_ * array_height_);
CHECK_OPENGL_ERROR("");

  glDisableClientState(GL_VERTEX_ARRAY);
CHECK_OPENGL_ERROR("");
  glDisableClientState(GL_COLOR_ARRAY);
CHECK_OPENGL_ERROR("");

#if PB
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif

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
//   cerr<<"Updating widget transform.\n";
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

