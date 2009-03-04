/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef GLSL_Shader_h
#define GLSL_Shader_h

#include <sci_gl.h>

namespace Kurt {


class GLSLShader {
public:
  GLSLShader(){}

  virtual ~GLSLShader(){}
  
  typedef enum {
    EVertexShader,
    EFragmentShader,
  } EShaderType;


  GLint getUniLoc(GLuint program, const GLchar *name);
  static void printShaderInfoLog(GLuint shader);
  static void printProgramInfoLog(GLuint program);

  static int shaderSize(char *fileName, EShaderType shaderType);
  static int readShader(char *fileName, EShaderType shaderType,
                        char *shaderText, int size);
  
  int read_shader_source(char *fileName, GLchar **vertexShader,
                       GLchar **fragmentShader);
  void install_shaders(const GLchar *particleVertex,
                      const GLchar *particleFragment);
  int compile_shaders();
  void create_program();
  void attach_objects();
  void bind_attribute_location(GLuint index, const GLchar *name);
  int link_program();


  void initialize_uniform( GLchar *name , GLfloat u0, 
                           GLfloat u1, GLfloat u2, GLfloat u3);
  void initialize_uniform( GLchar *name, GLfloat u0, GLfloat u1, GLfloat u2);
  void initialize_uniform( GLchar *name, GLfloat u0, GLfloat u1);
  void initialize_uniform( GLchar *name, GLfloat u0);
  void initialize_uniform( GLchar *name, GLint u0 );
  void initialize_uniform( GLchar *name, GLuint count,
                           GLboolean transpose, GLfloat *mat);
  
  
  GLuint program_object(){ return program_object_; }
  GLuint vertex_shader_object(){ return vertex_shader_object_; }
  GLuint fragment_shader_object(){ return fragment_shader_object_; }

protected:
  GLuint program_object_;
  GLuint vertex_shader_object_;
  GLuint fragment_shader_object_;

};

} // namespace Kurt

#endif // #ifndef GLSL_Shader_h
