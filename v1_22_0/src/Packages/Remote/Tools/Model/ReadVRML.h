#ifndef _READVRML_H_
#define _READVRML_H_

#include <Packages/Remote/Tools/Model/Object.h>

#include <stdlib.h>

#define PER_VERTEX_INDEXED 1
#define PER_FACE_INDEXED 2
#define PER_VERTEX 3
#define PER_FACE 4
#define OVERALL 5

namespace Remote {
// "Configuration" definitions for compiler
#define VRML_ATOF(a) (atof(a))
#define VRML_ATOI(a) (atoi(a))

// This object contains all the code necessary to parse an object from a VRML file.
struct YYObject : public Object
{
  Matrix44 Transform;
  Vector ScaleFac; // Cumulative scale factor
  int MaterialBinding, NormalBinding;
  float CreaseAngle;

  vector<int> MaterialIndices;
  vector<int> NormalIndices;
  vector<int> TexCoordIndices;
  vector<int> VertexIndices;

  YYObject()
  {
    CreaseAngle = 0.5; // Radians
    MaterialBinding = OVERALL;
    NormalBinding = PER_VERTEX_INDEXED;
    ScaleFac = Vector(1,1,1);
  }
};

// These are all given to BisonMe.y.
extern void s_Vertices(vector<Vector> *);
extern void s_Normals(vector<Vector> *);
extern void s_TexCoords(vector<Vector> *);
extern void s_AmbientColors(vector<Vector> *);
extern void s_DiffuseColors(vector<Vector> *);
extern void s_EmissiveColors(vector<Vector> *);
extern void s_Shininesses(vector<double> *);
extern void s_SpecularColors(vector<Vector> *);
extern void s_Transparencies(vector<double> *);

extern void s_CoordIndices(vector<int> *);
extern void s_MaterialIndices(vector<int> *);
extern void s_NormalIndices(vector<int> *);
extern void s_TexCoordIndices(vector<int> *);

extern void s_NormalBinding(int Binding);
extern void s_MaterialBinding(int Binding);

extern void s_CreaseAngle(float);
extern void s_OutputIndexedFaceSet();
extern void s_Texture2_filename(char *);
extern void s_Info(char *);
extern void s_Matrix(double *mat);
extern void s_ScaleFactor(Vector *);
extern void s_Separator_begin();
extern void s_Separator_end();

extern void yyerror(char *s);

} // End namespace Remote


#endif
