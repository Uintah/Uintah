#ifndef GFXSYS_FUTILS_INCLUDED // -*- C++ -*-
#define GFXSYS_FUTILS_INCLUDED

class tempFilename
{
    char *name;
public:
    tempFilename(char *dir="/tmp");
    ~tempFilename();

    const char *filename() const { return name; }
};


extern istream *pipe_input_stream(char *cmd, ...);


// GFXSYS_FUTILS_INCLUDED
#endif
