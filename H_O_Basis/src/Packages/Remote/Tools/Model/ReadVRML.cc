//////////////////////////////////////////////////////////////////////
// ReadVRML.cpp - Parse most of VRML 1.0 and return it as a
// vector<Object>.
// Originally by Bill Mark, June 1998.
// Hacked by David K. McAllister July 1999.
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>

#include <Packages/Remote/Tools/Math/Vector.h>
#include <Packages/Remote/Tools/Model/ReadVRML.h>
#include <Packages/Remote/Tools/Model/Model.h>
//#include <Packages/Remote/Tools/Model/BisonMe.h>

using namespace Remote::Tools;

extern int yyparse(void); // The function that bison makes.
extern int yynerrs; // Number of errors so far, maintained by bison.

namespace Remote {
//////////////////////////////////////////////////////////////////////
// Prototypes

extern int my_linecount; // number of lines read. Maintained by bison.
extern char my_linetext[]; // text of the current line, used in bison.

//////////////////////////////////////////////////////////////////////
// Global variables

FILE *InFile; // the flex code looks at this.

static vector<YYObject> Stack;	// The state stack.
Model *thisModel = NULL;

//////////////////////////////////////////////////////////////////////
// Callback routines from bison
//////////////////////////////////////////////////////////////////////

// Error handler for bison
void yyerror(char *s)
{
  fprintf(stderr, "* %s, on line %i:\n", s, my_linecount);
  fprintf(stderr, "%s\n", my_linetext); // str includes newline, add one
}

// Handle info field.
void s_Info(char *infostr)
{
  fprintf(stderr, "# VRML_INFO_FIELD: %s\n", infostr);
}

void s_Separator_begin()
{
  // Copy the top element.
  Stack.push_back(Stack.back());
  // fprintf(stderr, "Pushing the stack: %d\n", Stack.size());
}

void s_Separator_end()
{
  ASSERTERR(Stack.size() > 1, "ReadVRML: Internal Separator{} stack bug.");

  Stack.pop_back();
}

void s_CreaseAngle(float cr)
{
  YYObject &S = Stack.back();

  S.CreaseAngle = cr;
}

/* Matrix -- currently assumed to be for a MatrixTransform */
void s_Matrix(double *mat)
{
  Matrix44 Right(mat, true);

  YYObject &S = Stack.back();
  S.Transform *= Right;
}

/* Scale factor */
void s_ScaleFactor(Vector *s)
{
  YYObject &S = Stack.back();

  S.ScaleFac.x *= s->x;
  S.ScaleFac.y *= s->y;
  S.ScaleFac.z *= s->z;
}

// Load the specified texture.
void s_Texture2_filename(char *texFName)
{
  YYObject &S = Stack.back();

  S.TexInd = Model::TexDB.FindByName(texFName);

  if(S.TexInd < 0)
    {
      // Couldn't find it. Load it in.
      fprintf(stderr, "Loading %s\n", texFName);
      Image *Im = new Image(texFName);
      if(Im->size > 0)
	{
	  TexInfo T(texFName, Im, -1);
	  
	  Model::TexDB.TexList.push_back(T);
	  S.TexInd = Model::TexDB.TexList.size() - 1;
	}
      else
	fprintf(stderr, "Couldn't load texture %s\n", texFName);
    }
}

// Save x,y,z coordinates
void s_Vertices(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.verts = *V;
}

// Save normals
void s_Normals(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.normals = *V;
}

// Save texture coordinates
void s_TexCoords(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.texcoords = *V;
}

// Compute the average of all the colors listed
// since we don't store them per-vertex except diffuse.
static Vector AvgColor(vector<Vector> *V)
{
  int i;
  Vector A(0,0,0);
  for(i=0; i<V->size(); i++)
    A += (*V)[i];
  return A / double(i);
}

// Average ambient colors.
void s_AmbientColors(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.acolor = AvgColor(V);
  S.AColorValid = true;
}

// Save diffuse colors.
void s_DiffuseColors(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.dcolors = *V;
}

// Average emissive colors.
void s_EmissiveColors(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.ecolor = AvgColor(V);
  S.EColorValid = true;
}

// Average shininesses.
void s_Shininesses(vector<double> *V)
{
  int i;
  YYObject &S = Stack.back();
  S.shininess = 0;
  for(i=0; i<V->size(); i++)
    S.shininess += (*V)[i];
  S.shininess /= double(i);
  S.shininess *= 128.0; // Convert to OpenGL style.
  S.ShininessValid = true;
}

// Average specular colors.
void s_SpecularColors(vector<Vector> *V)
{
  YYObject &S = Stack.back();
  S.scolor = AvgColor(V);
  S.SColorValid = true;
}

void s_Transparencies(vector<double> *V)
{
  YYObject &S = Stack.back();
}

void s_CoordIndices(vector<int> *V)
{
  YYObject &S = Stack.back();
  S.VertexIndices = *V;
}

void s_MaterialIndices(vector<int> *V)
{
  YYObject &S = Stack.back();
  S.MaterialIndices = *V;
}

void s_NormalIndices(vector<int> *V)
{
  YYObject &S = Stack.back();
  S.NormalIndices = *V;
}

void s_TexCoordIndices(vector<int> *V)
{
  YYObject &S = Stack.back();
  S.TexCoordIndices = *V;
}

void s_MaterialBinding(int Binding)
{
  YYObject &S = Stack.back();
  S.MaterialBinding = Binding;
}

void s_NormalBinding(int Binding)
{
  YYObject &S = Stack.back();
  S.NormalBinding = Binding;
}

// Returns true on error.
static bool DoNormals(Object &B, YYObject &S, int i, int ii, int FaceNum)
{
  switch(S.NormalBinding)
    {
    case PER_VERTEX:
      // Just copy it.
      // XXX Is this right, or do I indirect by the vertex index?
      B.normals.push_back(S.normals[ii]);
      break;
    case PER_VERTEX_INDEXED:
      ASSERTERR(S.NormalIndices[i] >= 0 && S.NormalIndices[i] < S.normals.size(),
		"Normal index out of range");
      B.normals.push_back(S.normals[S.NormalIndices[i]]);
      break;
    case PER_FACE:
      // Replicate the normals.
      ASSERTERR(FaceNum >= 0 && FaceNum < S.normals.size(),
		"Normal face num out of range");
      B.normals.push_back(S.normals[FaceNum]);
      break;
    case PER_FACE_INDEXED:
      {
	ASSERTERR(FaceNum >= 0 && FaceNum < S.NormalIndices.size(),
		  "Normal face num out of range");
	int f = S.NormalIndices[FaceNum];
	ASSERTERR(f >= 0 && f < S.normals.size(),
		  "Normal face index out of range");
	B.normals.push_back(S.normals[f]);
	break;
      }
    case OVERALL:
      break;
    default:
      fprintf(stderr, "Unknown normal binding.\n");
      return true;
    }

  return false;
}

// Returns true on error.
static bool DoColors(Object &B, YYObject &S, int i, int ii, int FaceNum, bool doalpha)
{
  switch(S.MaterialBinding)
    {
    case PER_VERTEX:
      // Just copy it.
      // XXX Is this right, or do I indirect by the vertex index?
      // fprintf(stderr, "verts:%d, vi:%d, dc:%d, i:%d\n", S.verts.size(), S.VertexIndices.size(),
      // S.dcolors.size(), ii);

      ASSERTERR(ii >= 0 && ii < S.dcolors.size(), "Dcolor out of range");

      B.dcolors.push_back(S.dcolors[ii]);
      if(doalpha)
	{
	  ASSERTERR(ii >= 0 && ii < S.alpha.size(), "Alpha out of range");
	  B.alpha.push_back(1-S.alpha[ii]);
	}
      break;
    case PER_VERTEX_INDEXED:
      ASSERTERR(S.MaterialIndices[i] >= 0 && S.MaterialIndices[i] < S.dcolors.size(),
		"Dcolor index out of range");
      B.dcolors.push_back(S.dcolors[S.MaterialIndices[i]]);
      if(doalpha)
	{
	  ASSERTERR(i >= 0 && i < S.alpha.size(), "Alpha index out of range");
	  B.alpha.push_back(1-S.alpha[S.MaterialIndices[i]]);
	}
      break;
    case PER_FACE:
      // Replicate the dcolors.
      ASSERTERR(FaceNum >= 0 && FaceNum < S.dcolors.size(),
		"Dcolor face num out of range");
      B.dcolors.push_back(S.dcolors[FaceNum]);
      if(doalpha)
	B.alpha.push_back(1-S.alpha[FaceNum]);
      break;
    case PER_FACE_INDEXED:
      {
	ASSERTERR(FaceNum >= 0 && FaceNum < S.MaterialIndices.size(),
		  "Dcolor face num out of range");
	int f = S.MaterialIndices[FaceNum];
	ASSERTERR(f >= 0 && f < S.dcolors.size(),
		  "Dcolor face index out of range");
	B.dcolors.push_back(S.dcolors[f]);
	if(doalpha)
	  B.alpha.push_back(1-S.alpha[f]);
	break;
      }
    case OVERALL:
      break;
    default:
      fprintf(stderr, "Unknown dcolor binding.\n");
      return true;
    }
  return false;
}

static void DoVertex(Object &B, YYObject &S, int i, int ii, int FaceNum, bool doalpha, bool docolors, bool donormals, Matrix44 &Tran)
{
  ASSERTERR(S.VertexIndices[i] < S.verts.size() && S.VertexIndices[i] >= 0,
	    "Vertex index out of range.");
  Vector V = S.verts[S.VertexIndices[i]];
  Vector Vp = Tran * V;
  
  B.Box += Vp;
  B.verts.push_back(Vp);
  
  // We should also handle synthesizing texcoords.
  if(S.TexCoordIndices.size() > 0)
    {
      ASSERTERR(S.TexCoordIndices[i] >= 0 && S.TexCoordIndices[i] < S.texcoords.size(),
		"texcoord index out of range");
      B.texcoords.push_back(S.texcoords[S.TexCoordIndices[i]]);
    }
  
  if(donormals)
    if(DoNormals(B, S, i, ii, FaceNum)) return;
  
  // Handle diffuse color.
  if(docolors)
    if(DoColors(B, S, i, ii, FaceNum, doalpha)) return;
}

// Touches up the Object to convert from the file's format to the official
// semantics of Object and adds it to the Model.
void s_OutputIndexedFaceSet()
{
  YYObject &S = Stack.back();
  if(S.VertexIndices.size() < 1)
    {
      fprintf(stderr, "No vertex indices.\n");
      return;
    }

  // This is mostly for copying state. The rest of the function
  // re-copies the vertex values.
  Object B = (Object) S;

  B.PrimType = L_TRIANGLES;

  // Make sure the texture coord index list is the right size.
  if(S.TexCoordIndices.size() > 0)
    ASSERTERR(S.TexCoordIndices.size() == S.VertexIndices.size(),
	      "Wrong num. of texcoord indices.");

  B.verts.clear();
  B.texcoords.clear();
  // Can have a single global normal or build one per vertex.
  bool donormals = false;
  if(S.normals.size() > 1)
    {
      donormals = true;
      B.normals.clear();
      if(S.NormalBinding == OVERALL)
	B.normals.push_back(S.normals[0]);
    }

  if(S.alpha.size() > 0)
    ASSERTERR(S.alpha.size() == S.dcolors.size(), "Inconsistent alpha list.");
  bool docolors = false, doalpha = false;
  if(S.dcolors.size() > 1)
    {
      if(S.alpha.size() > 1)
	doalpha = true;

      docolors = true;
      B.alpha.clear();
      B.dcolors.clear();
      if(S.MaterialBinding == OVERALL)
	B.dcolors.push_back(S.dcolors[0]);
    }
  else if(B.alpha.size() == 1)
    B.alpha[0] = 1 - B.alpha[0];

  Matrix44 Tran(S.Transform);
  Tran.Scale(S.ScaleFac);

  int FaceNum = 0, ci = 0, vertsThisPoly = 0;
  for(int i=0; i<S.VertexIndices.size(); i++)
    {
      if(S.VertexIndices[i] != -1)
	{
	  if(vertsThisPoly >= 3)
	    {
	      // Handle arbitrary convex polygons.
	      DoVertex(B, S, i-vertsThisPoly, ci-vertsThisPoly, FaceNum, doalpha, docolors, donormals, Tran);
	      DoVertex(B, S, i-1, ci-1, FaceNum, doalpha, docolors, donormals, Tran);
	    }

	  DoVertex(B, S, i, ci, FaceNum, doalpha, docolors, donormals, Tran);
	  
	  ci++;
	  vertsThisPoly++;
	}
      else
	{
	  FaceNum++;
	  vertsThisPoly = 0;
	}
    }

  // Generate normals if we don't have enough.
  B.creaseAngle = S.CreaseAngle;

  // THe final result of this function is to add this object to the
  // list of objects that will be returned to the user.
  thisModel->Objs.push_back(B);
  
  // Update bounding box of the whole model.
  thisModel->Box += B.Box;
}

// Returns false on success.
bool Model::ReadVRML(const char *fname)
{
  InFile = fopen(fname, "r");
  ASSERTERR(InFile, "Error opening input file");

  // Read first line to verify that it's VRML 1.0 Best to do this
  // here, since it's hard to make lex'er distinguish between this
  // first line and a comment
  char tmpbuf[256];
  char firstline[] = "#VRML V1.0 ascii";
  if(!fgets(tmpbuf, 256, InFile) || strncmp(firstline, tmpbuf, strlen(firstline)))
    {
      fprintf(stderr, "Input file is not VRML 1.0\n");
      fclose(InFile);
      return true;
    }

  // ObjectSet is a global pointer into our list of objects which
  // allows the global callback routines to store objects in our model.
  Objs.clear();
  thisModel = this;
  Stack.clear();
  YYObject First;
  Stack.push_back(First);

  fprintf(stderr, "Starting parse.\n");
  int parseret = yyparse();
  fclose(InFile);

  if(parseret)
    {
      fprintf(stderr, "Parsing terminated with %i errors.\n", yynerrs);
      return true;
    }
  else
    fprintf(stderr, "Parsing completed with no errors.\n");

  // Check that state stack is balanced -- should always be true.
  ASSERTERR(Stack.size()==1, "ReadVRML:: Internal error -- Stack depth bad.");

  return false;
}
} // End namespace Remote


