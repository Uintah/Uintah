#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <fstream.h>

#include <gfx/std.h>
#include <gfx/sys/futils.h>
#include <gfx/SMF/smf.h>
#include <gfx/SMF/smfstate.h>

inline int streq(const char *a,const char *b) { return strcmp(a,b)==0; }

float SMF_version = 1.0;
char *SMF_version_string = "1.0";
char *SMF_source_revision = "$Id$";



int SMF_Model::in_Face(const buffer<int>&)
{
    fatal_error("SMF: Arbitrary face definitions not supported.");
    return False;
}

int SMF_Model::in_Unparsed(const char *, string_buffer&)
{
    return False;
}

int SMF_Model::in_VNormal(const Vec3&) { return True; }
int SMF_Model::in_VColor(const Vec3&) { return True; }
int SMF_Model::in_FColor(const Vec3&) { return True; }

int SMF_Model::note_Vertices(int) { return True; }
int SMF_Model::note_Faces(int) { return True; }
int SMF_Model::note_BBox(const Vec3&, const Vec3&) { return True; }
int SMF_Model::note_BSphere(const Vec3&, real) { return True; }
int SMF_Model::note_PXform(const Mat4&) { return True; }
int SMF_Model::note_MXform(const Mat4&) { return True; }
int SMF_Model::note_Unparsed(const char *,string_buffer&) { return True; }



SMF_Reader::cmd_entry SMF_Reader::read_cmds[] = {
    { "v", &SMF_Reader::vertex },
    { ":vn", &SMF_Reader::v_normal },
    { ":vc", &SMF_Reader::v_color },
    { ":fc", &SMF_Reader::f_color },
    { "t", &SMF_Reader::face },
    { "f", &SMF_Reader::face },

    { "begin", &SMF_Reader::begin },
    { "end", &SMF_Reader::end },
    { "set", &SMF_Reader::set },
    { "inc", &SMF_Reader::inc },
    { "dec", &SMF_Reader::dec },

    { "mmult", &SMF_Reader::mload },
    { "mload", &SMF_Reader::mmult },
    { "trans", &SMF_Reader::trans },
    { "scale", &SMF_Reader::scale },
    { "rot", &SMF_Reader::rot },

    { NULL, NULL }
};

static Mat4 mat_from_args(string_buffer& argv)
{
    Mat4 M;

    int n=0, i, j;
    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	    M(i,j) = atof(argv(n++));

    return M;
}

static
void bad_annotation(char *cmd)
{
    cerr << "SMF: Malformed annotation ["<< cmd << "]" << endl;
}

void SMF_Reader::annotation(char *cmd, string_buffer& argv)
{
    // Skip over the '#$' prefix
    cmd+=2;

    if( streq(cmd,"SMF") ) {
	if( atof(argv(0)) != SMF_version )
	    cerr << "SMF: Version mismatch ("
		 << argv(0) << " instead of "
		 << SMF_version_string << ")" << endl;
    }
    else if( streq(cmd,"vertices") )
    {
	if( argv.length() == 1 )
	    model->note_Vertices(atoi(argv(0)));
	else
	    bad_annotation(cmd);
    }
    else if( streq(cmd, "faces") )
    {
	if( argv.length() == 1 )
	    model->note_Faces(atoi(argv(0)));
	else
	    bad_annotation(cmd);

    }
    else if( streq(cmd, "BBox") )
    {
    }
    else if( streq(cmd, "BSphere") )
    {
    }
    else if( streq(cmd, "PXform") )
    {
	if( argv.length() == 16 )
	    model->note_PXform(mat_from_args(argv));
	else
	    bad_annotation(cmd);
    }
    else if( streq(cmd, "MXform") )
    {
	if( argv.length() == 16 )
	    model->note_MXform(mat_from_args(argv));
	else
	    bad_annotation(cmd);
    }
    else
	model->note_Unparsed(cmd, argv);
}

void SMF_Reader::begin(string_buffer&) { state = new SMF_State(ivar,state); }
void SMF_Reader::end(string_buffer&)
{
    SMF_State *old = state;
    state=state->pop();
    delete old;
}

void SMF_Reader::set(string_buffer& argv)
{
    state->set(argv);
}

void SMF_Reader::inc(string_buffer&)
{
    cerr << "SMF: INC not yet implemented." << endl;
}

void SMF_Reader::dec(string_buffer&)
{
    cerr << "SMF: DEC not yet implemented." << endl;
}

void SMF_Reader::trans(string_buffer& argv)
{
    Mat4 M = Mat4::trans(atof(argv(0)), atof(argv(1)), atof(argv(2)));
    state->mmult(M);
}

void SMF_Reader::scale(string_buffer& argv)
{
    Mat4 M = Mat4::scale(atof(argv(0)), atof(argv(1)), atof(argv(2)));
    state->mmult(M);
}

void SMF_Reader::rot(string_buffer& argv)
{
    switch( argv(0)[0] )
    {
    case 'x':
	state->mmult(Mat4::xrot(atof(argv(1)) * M_PI/180.0));
	break;
    case 'y':
	state->mmult(Mat4::yrot(atof(argv(1)) * M_PI/180.0));
	break;
    case 'z':
	state->mmult(Mat4::zrot(atof(argv(1)) * M_PI/180.0));
	break;
    default:
	cerr << "SMF: Malformed rotation command" << endl;
	break;
    }
}

void SMF_Reader::mmult(string_buffer& argv)
{
    state->mmult(mat_from_args(argv));
}

void SMF_Reader::mload(string_buffer& argv)
{
    state->mload(mat_from_args(argv));
}


void SMF_Reader::vertex(string_buffer& argv)
{
    Vec3 v;

    v[X] = atof(argv(0));
    v[Y] = atof(argv(1));
    v[Z] = atof(argv(2));

    state->vertex(v);
    ivar.next_vertex++;

    model->in_Vertex(v);
}

void SMF_Reader::v_normal(string_buffer& argv)
{
    Vec3 n;

    n[X] = atof(argv(0));
    n[Y] = atof(argv(1));
    n[Z] = atof(argv(2));

    state->normal(n);

    model->in_VNormal(n);
}

void SMF_Reader::v_color(string_buffer& argv)
{
    Vec3 c;

    c[X] = atof(argv(0));
    c[Y] = atof(argv(1));
    c[Z] = atof(argv(2));

    model->in_VColor(c);
}

void SMF_Reader::f_color(string_buffer& argv)
{
    Vec3 c;

    c[X] = atof(argv(0));
    c[Y] = atof(argv(1));
    c[Z] = atof(argv(2));

    model->in_FColor(c);
}

void SMF_Reader::face(string_buffer& argv)
{
    buffer<int> verts(3);

    for(int i=0; i<argv.length(); i++)
	verts.add(atoi(argv(i)));

    state->face(verts, ivar);
    ivar.next_face++;

    if( verts.length() == 3 )
	model->in_Face(verts(0), verts(1), verts(2));
    else
	model->in_Face(verts);
}



void SMF_Reader::parse_line(char *line)
{
    char *cmd,*s;
    string_buffer argv(16);

    while( *line==' ' || *line=='\t' ) line++;  // skip initial white space

    // Ignore empty lines
    if( line[0]=='\n' || line[0]=='\0' ) return;

    // Ignore comments
    if( line[0]=='#' && line[1]!='$' ) return;

    //
    // First, split the line into tokens
    cmd = strtok(line, " \t\n");

    while( (s=strtok(NULL, " \t\n")) )
	argv.add(s);

    //
    // Figure out what command it is and execute it
    if( cmd[0]=='#' && cmd[1]=='$' )
	annotation(cmd,argv);
    else
    {
	cmd_entry *entry = &read_cmds[0];
	bool handled = False;

	while( entry->name && !handled )
	    if( streq(entry->name, cmd) )
	    {
		(this->*(entry->cmd))(argv);
		handled = True;
	    }
	    else
		entry++;

	if( !handled && !model->in_Unparsed(cmd, argv) )
	{
	    // Invalid command:
	    cerr << "SMF: Illegal command [" << cmd << "]" << endl;
	    exit(1);
	}
    }
}


void SMF_Reader::init(istream *i)
{
    ivar.next_face = 1;
    ivar.next_vertex = 1;

    in_p = i;
    state = new SMF_State(ivar);
    line = new char[SMF_MAXLINE];
}

SMF_Reader::SMF_Reader(istream& i)
{
    init(&i);
}

SMF_Reader::SMF_Reader(char *filename)
{
    istream *model_stream;

    if( filename[0]=='-' && filename[1]=='\0' )
	model_stream = &cin;
    else
    {
	char suffix = filename[strlen(filename)-1];
	if( suffix=='Z' || suffix=='z' )
	    model_stream = pipe_input_stream("gzip","-dc",filename,NULL);
	else
	    model_stream = new ifstream(filename);

	if( !model_stream->good() )
	{
	    cerr << "SMF: Unable to open input file '"
		 << filename << "'" << endl;
	    exit(1);
	}
    }

    init(model_stream);
}

SMF_Reader::~SMF_Reader()
{
    //!!FILL ME IN
}


void SMF_Reader::read(SMF_Model *m)
{
    model = m;

    istream& in = *in_p;

    while( !in.eof() )
    {
	if( in.getline(line, SMF_MAXLINE, '\n').good() )
	    parse_line(line);
    }
}
