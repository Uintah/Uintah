#ifndef GFX_SMF_INCLUDED // -*- C++ -*-
#define GFX_SMF_INCLUDED

/************************************************************************

  SMF file parser.

  $Id$

 ************************************************************************/

#include <iostream.h>
#include <gfx/tools/Buffer.h>
#include <gfx/math/Vec3.h>
#include <gfx/math/Mat4.h>

#define SMF_MAXLINE 4096

extern float SMF_version;
extern char *SMF_version_string;
extern char *SMF_source_revision;

typedef buffer<char *> string_buffer;


class SMF_Model
{
public:

    ////////////////////////////////////////////////////////////////////////
    //
    // SMF input methods
    //

    //
    // These are the REQUIRED methods
    virtual int in_Vertex(const Vec3&) = 0;
    virtual int in_Face(int v1, int v2, int v3) = 0;

    //
    // By default, arbitrary faces are flagged as errors
    virtual int in_Face(const buffer<int> &);
    // 
    // as are unknown commands
    virtual int in_Unparsed(const char *, string_buffer&);

    //
    // These methods are optional.  By default, they'll be ignored
    virtual int in_VColor(const Vec3&);
    virtual int in_VNormal(const Vec3&);
    virtual int in_FColor(const Vec3&);

    virtual int note_Vertices(int);
    virtual int note_Faces(int);
    virtual int note_BBox(const Vec3& min, const Vec3& max);
    virtual int note_BSphere(const Vec3&, real);
    virtual int note_PXform(const Mat4&);
    virtual int note_MXform(const Mat4&);
    virtual int note_Unparsed(const char *,string_buffer&);

    ////////////////////////////////////////////////////////////////////////
    //
    // SMF output methods
    //

//     virtual void annotate_header(FILE *);
//     virtual void annotate_Vertex(FILE *,int);
//     virtual void annotate_Face(FILE *,int);
};

class SMF_State;

//
// Internal SMF variables
// (not accessible via 'set' commands)
//
struct SMF_ivars
{
    int next_vertex;
    int next_face;
};

class SMF_Reader
{
public:
    typedef void (SMF_Reader::*read_cmd)(string_buffer& argv);
    struct cmd_entry { char *name; read_cmd cmd; };

private:
    char *line;
    SMF_State *state;
    SMF_ivars ivar;
    static cmd_entry read_cmds[];

    void init(istream *);

protected:


    void annotation(char *cmd, string_buffer& argv);

    void vertex(string_buffer&);
    void v_normal(string_buffer&);
    void v_color(string_buffer&);
    void f_color(string_buffer&);
    void face(string_buffer&);

    void begin(string_buffer&);
    void end(string_buffer&);
    void set(string_buffer&);
    void inc(string_buffer&);
    void dec(string_buffer&);

    void trans(string_buffer&);
    void scale(string_buffer&);
    void rot(string_buffer&);
    void mmult(string_buffer&);
    void mload(string_buffer&);

    istream *in_p;
    SMF_Model *model;

public:

    SMF_Reader(istream&);
    SMF_Reader(char *filename);
    ~SMF_Reader();

    void read(SMF_Model *);

    void parse_line(char *line);
};


// GFX_SMF_INCLUDED
#endif
