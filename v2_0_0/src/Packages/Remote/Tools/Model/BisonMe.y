%{
/*
 * BisonMe.y -- bison (yacc-like) parser description for VRML1.0.
 * Mostly by Bill Mark, 1998.
 * Enhanced by DaveMc, 1999.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Packages/Remote/Tools/Math/Vector.h>
#include <Packages/Remote/Tools/Model/ReadVRML.h>
using namespace Remote::Tools;

int yylex();

/* define variables for return of actual semantic values from lexer
 * note that this is a kludge to avoid having to do a %union in bison,
 * which I discovered creates a large number of annoyances.
 */

/* make calls to yyerror produce verbose error messages (at least for now) */
#define YYDEBUG 1
#define YYERROR_VERBOSE 1

/* Prototypes */
char *tokentext(void);  /* return token text */

%}

%union {
  int		none; /* Used to flag values w/o type */
  int		ival;
  double	fval;
  char		*sval;
  double	*mptr;
  Vector	*vec;
  vector<int>	*ibuf;
  vector<double> *fbuf;
  vector<Vector> *vbuf;
}

%{
/* error */
%}
%token <none> T_ERROR

%{
/* literals */
%}
%token <ival> T_INT
%token <fval> T_FLOAT
%token <sval> T_STRING

%{
/* Keywords.  All begin with TR_ */
%}
%token <none> TR_DEF
%token <none> TR_Separator
%token <none> TR_name
%token <none> TR_map
%token <none> TR_NONE
%token <none> TR_Info
%token <none> TR_string
%token <none> TR_PerspectiveCamera
%token <none> TR_position
%token <none> TR_orientation
%token <none> TR_focalDistance
%token <none> TR_heightAngle
%token <none> TR_Scale
%token <none> TR_scaleFactor
%token <none> TR_MaterialBinding
%token <none> TR_NormalBinding
%token <none> TR_value
%token <none> TR_OVERALL
%token <none> TR_PER_FACE
%token <none> TR_PER_FACE_INDEXED
%token <none> TR_PER_VERTEX
%token <none> TR_PER_VERTEX_INDEXED
%token <none> TR_ShapeHints
%token <none> TR_vertexOrdering
%token <none> TR_COUNTERCLOCKWISE
%token <none> TR_shapeType
%token <none> TR_SOLID
%token <none> TR_faceType
%token <none> TR_CONVEX
%token <none> TR_creaseAngle
%token <none> TR_MatrixTransform
%token <none> TR_matrix
%token <none> TR_renderCulling
%token <none> TR_ON
%token <none> TR_OFF
%token <none> TR_AUTO
%token <none> TR_Texture2
%token <none> TR_filename
%token <none> TR_Coordinate3
%token <none> TR_point
%token <none> TR_Normal
%token <none> TR_vector
%token <none> TR_Material
%token <none> TR_ambientColor
%token <none> TR_specularColor
%token <none> TR_emissiveColor
%token <none> TR_diffuseColor
%token <none> TR_shininess
%token <none> TR_transparency
%token <none> TR_TextureCoordinate2
%token <none> TR_IndexedFaceSet
%token <none> TR_coordIndex
%token <none> TR_materialIndex
%token <none> TR_normalIndex
%token <none> TR_textureCoordIndex

%type <ival> rcopt
%type <mptr> matrix4x4

%type <vbuf> c3_field
%type <vbuf> v3_field
%type <vbuf> acolor
%type <vbuf> dcolor
%type <vbuf> ecolor
%type <vbuf> scolor
%type <fbuf> shine
%type <fbuf> transp
%type <vbuf> tc2_field
%type <ival> mb_field
%type <ival> nb_field

%type <vec>  triple
%type <vbuf> triples
%type <vbuf> onetriple
%type <vbuf> rtriples
%type <vec>  double
%type <vbuf> doubles
%type <vbuf> onedouble
%type <vbuf> rdoubles
%type <ival> isingle
%type <ibuf> isingles
%type <ibuf> risingles
%type <fval> fnum
%type <fbuf> fsingles
%type <fbuf> onefsingle
%type <fbuf> rfsingles

%%

vrmlfile	: node
		;

node		: rnode						{}
		| TR_DEF T_STRING rnode				{}
		;

rnode		: TR_Separator '{'				{s_Separator_begin();}
                  separator_fields nodes '}'			{s_Separator_end();}
                | TR_Info '{' info_fields '}'			{}
		| TR_PerspectiveCamera '{' pc_fields '}'        {}
		| TR_Scale '{' sf_field '}'			{}
		| TR_MaterialBinding '{' mb_field '}'		{s_MaterialBinding($3);}
		| TR_NormalBinding '{' nb_field '}'		{s_NormalBinding($3);}
		| TR_ShapeHints '{' sh_fields '}'		{}
		| TR_MatrixTransform '{' mt_field '}'		{}
		| TR_Texture2 '{' t2_field '}'			{}
		| TR_Coordinate3 '{' c3_field '}'		{s_Vertices($3);}
		| TR_Material '{' mat_fields '}'		{}
		| TR_Normal '{' v3_field '}'			{s_Normals($3);}
		| TR_TextureCoordinate2 '{' tc2_field '}'	{s_TexCoords($3);}
		| TR_IndexedFaceSet '{' ifs_fields '}'		{s_OutputIndexedFaceSet();}
		;

nodes		: nodes node
		|
		;

separator_fields: separator_fields separator_field
		|
		;

separator_field	: TR_renderCulling rcopt			{}
		| TR_name T_STRING				{/* Really part of WWWAnchor*/}
		| TR_map TR_NONE				{/* Really part of WWWAnchor*/}
		;

rcopt		: TR_ON		{$$ = TR_ON;}
		| TR_OFF	{$$ = TR_OFF;}
		| TR_AUTO	{$$ = TR_AUTO;}
		;

info_fields	: info_fields info_field
		|
		;

info_field	: TR_string T_STRING {s_Info($2);}
		;

pc_fields	: pc_fields pc_field
		|
		;

pc_field	: TR_position      fnum fnum fnum		{}
		| TR_orientation   fnum fnum fnum fnum		{}
		| TR_focalDistance fnum				{}
		| TR_heightAngle   fnum				{}
		;

sf_field	: TR_scaleFactor triple				{s_ScaleFactor($2);}
		;

mb_field	: TR_value TR_PER_VERTEX_INDEXED		{$$ = PER_VERTEX_INDEXED}
		| TR_value TR_PER_FACE_INDEXED			{$$ = PER_FACE_INDEXED}
		| TR_value TR_OVERALL				{$$ = OVERALL}
		| TR_value TR_PER_FACE				{$$ = PER_FACE}
		| TR_value TR_PER_VERTEX			{$$ = PER_VERTEX}
		;

nb_field	: TR_value TR_PER_VERTEX_INDEXED		{$$ = PER_VERTEX_INDEXED}
		| TR_value TR_PER_FACE_INDEXED			{$$ = PER_FACE_INDEXED}
		| TR_value TR_OVERALL				{$$ = OVERALL}
		| TR_value TR_PER_FACE				{$$ = PER_FACE}
		| TR_value TR_PER_VERTEX			{$$ = PER_VERTEX}
		;

sh_fields	: sh_fields sh_field
		|
		;

sh_field	: TR_vertexOrdering TR_COUNTERCLOCKWISE		{}
		| TR_shapeType	    TR_SOLID			{}
		| TR_faceType	    TR_CONVEX			{}
		| TR_creaseAngle    fnum			{s_CreaseAngle($2);}
		;

mt_field	: TR_matrix matrix4x4		   {s_Matrix($2); delete [] $2;}
		;

matrix4x4	: fnum fnum fnum fnum
	          fnum fnum fnum fnum
	          fnum fnum fnum fnum
	          fnum fnum fnum fnum
                    {$$ = new double[16];
	             ASSERTERR($$, "new matrix failed");
		     $$[0] = $1;   $$[1] = $2;   $$[2] = $3;   $$[3] = $4;
                     $$[4] = $5;   $$[5] = $6;   $$[6] = $7;   $$[7] = $8;
                     $$[8] = $9;   $$[9] = $10;  $$[10] = $11; $$[11] = $12;
                     $$[12] = $13; $$[13] = $14; $$[14] = $15; $$[15] = $16;}
		;

t2_field	: TR_filename T_STRING {s_Texture2_filename($2);}
		;

c3_field	: TR_point '[' triples ']'		{$$ = $3;}
		| TR_point  onetriple			{$$ = $2;}
		;

v3_field	: TR_vector '[' triples ']'		{$$ = $3;}
		| TR_vector  onetriple			{$$ = $2;}
		;

tc2_field	: TR_point '[' doubles ']'		{$$ = $3;}
		| TR_point onedouble			{$$ = $2;}
		;

ifs_fields	: ifs_field
		| ifs_fields ifs_field
		;

ifs_field	: TR_coordIndex '[' isingles ']'	{s_CoordIndices($3);}
		| TR_materialIndex '[' isingles ']'	{s_MaterialIndices($3);}
		| TR_normalIndex '[' isingles ']'	{s_NormalIndices($3);}
		| TR_textureCoordIndex '[' isingles ']'	{s_TexCoordIndices($3);}
		;

mat_fields	: mat_fields mat_field
		|
		;

mat_field	: TR_ambientColor  acolor		{s_AmbientColors($2);}
		| TR_diffuseColor  dcolor		{s_DiffuseColors($2);}
		| TR_emissiveColor ecolor		{s_EmissiveColors($2);}
		| TR_specularColor scolor		{s_SpecularColors($2);}
		| TR_shininess	   shine		{s_Shininesses($2);}
		| TR_transparency  transp		{s_Transparencies($2);}
		;

acolor		: '[' triples ']'			{$$ = $2;}
		| onetriple				{$$ = $1;}
		;

dcolor		: '[' triples ']'			{$$ = $2;}
		| onetriple				{$$ = $1;}
		;

ecolor		: '[' triples ']'			{$$ = $2;}
		| onetriple				{$$ = $1;}
		;

scolor		: '[' triples ']'			{$$ = $2;}
		| onetriple				{$$ = $1;}
		;

shine		: '[' fsingles ']'			{$$ = $2;}
		| onefsingle				{$$ = $1;}
		;

transp		: '[' fsingles ']'			{$$ = $2;}
		| onefsingle				{$$ = $1;}
		;

triples		: rtriples			{$$ = $1}
		| rtriples ','			{$$ = $1}
		;

rtriples	: triple			{$$=new vector<Vector>; $$->push_back(*$1); delete $1;}
		| rtriples ',' triple		{$$ = $1; $$->push_back(*$3); delete $3;}
		;

onetriple	: fnum fnum fnum		{$$=new vector<Vector>; $$->push_back(Vector($1, $2, $3));}
		;

triple		: fnum fnum fnum		{$$ = new Vector($1, $2, $3);}
		;

doubles		: rdoubles			{$$ = $1}
		| rdoubles ','			{$$ = $1}
		;

rdoubles	: double			{$$=new vector<Vector>; $$->push_back(*$1); delete $1;}
		| rdoubles ',' double		{$$ = $1; $$->push_back(*$3); delete $3;}
		;

onedouble	: fnum fnum			{$$=new vector<Vector>; $$->push_back(Vector($1, $2, 0));}
		;

double		: fnum fnum			{$$ = new Vector($1, $2, 0);}
		;

isingles	: risingles			{$$ = $1}
		| risingles ','			{$$ = $1}
		;

risingles	: isingle			{$$ = new vector<int>; $$->push_back($1);}
		| risingles ',' isingle		{$$ = $1; $$->push_back($3);}
		;

isingle		: T_INT				{$$ = $1;}

fsingles	: rfsingles			{$$ = $1}
		| rfsingles ','			{$$ = $1}
		;

rfsingles	: fnum				{$$ = new vector<double>; $$->push_back($1);}
		| rfsingles ',' fnum		{$$ = $1; $$->push_back($3);}
		;

onefsingle	: fnum				{$$ = new vector<double>; $$->push_back($1);}
		;

fnum		: T_INT				{$$ = (double) $1;}
                | T_FLOAT			{$$ = $1;}
		;

%%
/* get yylex() from FlexMe.cpp */
#include <Packages/Remote/Tools/Model/FlexMe.cc>

