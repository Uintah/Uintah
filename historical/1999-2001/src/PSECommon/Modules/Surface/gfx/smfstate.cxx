#include <iostream.h>
#include <string.h>

#include <gfx/std.h>
#include <gfx/SMF/smf.h>
#include <gfx/SMF/smfstate.h>

inline int streq(const char *a,const char *b) { return strcmp(a,b)==0; }


SMF_State::SMF_State(const SMF_ivars& ivar, SMF_State *link)
{
    next = link;
    first_vertex = ivar.next_vertex;
    if( next )
    {
	vertex_correction = next->vertex_correction;
	xform = next->xform;
    }
    else
    {
	vertex_correction = 0;
	xform = Mat4::identity;
    }

}

void SMF_State::vertex(Vec3& v)
{
    Vec4 v2 = xform * Vec4(v,1);

    v[X] = v2[X]/v2[W];
    v[Y] = v2[Y]/v2[W];
    v[Z] = v2[Z]/v2[W];
}

void SMF_State::normal(Vec3& v)
{
    Vec4 v2 = xform * Vec4(v,0);

//     v[X] = v2[X]/v2[W];
//     v[Y] = v2[Y]/v2[W];
//     v[Z] = v2[Z]/v2[W];

    v[X] = v2[X];
    v[Y] = v2[Y];
    v[Z] = v2[Z];
}

void SMF_State::face(buffer<int>& verts, const SMF_ivars& ivar)
{
    for(int i=0; i<verts.length(); i++)
    {
	if( verts(i) < 0 )
	    verts(i) += ivar.next_vertex;
	else
	    verts(i) += vertex_correction + (first_vertex - 1);
    }
}

void SMF_State::set(string_buffer& argv)
{
    char *cmd = argv(0);

    if( streq(cmd, "vertex_correction") )
	vertex_correction = atoi(argv(1));
}

void SMF_State::mmult(const Mat4& M)
{
    xform = xform * M;
}

void SMF_State::mload(const Mat4& M)
{
    xform = M;
}
