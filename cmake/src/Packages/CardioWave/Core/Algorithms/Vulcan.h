/* This code is adapted from Chris Penland's Vulcan code in Matlab 
 Converted to C by Joe Tranquillo July 2001 */

typedef struct 
{
  float x,y,z;
  float volume;
  float sxx,sxy,sxz,syy,syz,szz;
}
VERTEX;

typedef struct 
{
	int vtx[8];
}
VOLUME_ELEMENT;

typedef struct
{
	int numvtx;
	int numelement;
	VERTEX* vtx;
	VOLUME_ELEMENT* elements;
}
MESH;

void determine_endianness ();
void init();
void read_mesh();
void read_conductivity(MESH* mesh);
void compute_volumes(MESH* mesh, const char* fname);
void compute_matrix(MESH* mesh, const char *fname);
void dump_vis(MESH* mesh, const char *fname);
void jacobian(float xi, float eta, float zeta,float Xbox[],float Ybox[],float Zbox[],float Jac[][3]);
void trial(float xi, float eta, float zeta, char ch, float dN[][3]);
void fvelement(float Xbox[],float Ybox[],float Zbox[],float lmatrix[][8],float D[][9]);
void normals(float Xbox[],float Ybox[],float Zbox[],float D[][9],float normal[][12]);
