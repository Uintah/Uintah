#ifndef GFXSMF_STATE_INCLUDED // -*- C++ -*-
#define GFXSMF_STATE_INCLUDED

class SMF_State
{
private:
    SMF_State *next;

    //
    // Standard state variables
    int first_vertex;
    int vertex_correction;
    Mat4 xform;

public:
    SMF_State(const SMF_ivars& ivar,SMF_State *link=NULL);
    SMF_State *pop() { return next; }

    void set(string_buffer& argv);
    void inc(const char *var, int delta=1);
    void dec(const char *var, int delta=1);

    void mmult(const Mat4&);
    void mload(const Mat4&);

    void vertex(Vec3&);
    void normal(Vec3&);
    void face(buffer<int>&, const SMF_ivars& ivar);
};


// GFXSMF_STATE_INCLUDED
#endif
