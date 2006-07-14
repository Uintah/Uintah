#ifndef GLSL_Shader_h
#define GLSL_Shader_h

#include <sci_gl.h>

namespace Uintah {


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

} // namespace Uintah

#endif // #ifndef GLSL_Shader_h
