#ifndef NAUTILUS_NPRIM_INCLUDED // -*- C++ -*-
#define NAUTILUS_NPRIM_INCLUDED

/************************************************************************

  Common code for modeling primitives.  These are the interfaces that
  most modeling primitives are expected to inherit.
  $Id$

 ************************************************************************/

class NPrim
{
public:
    int uniqID;

    inline bool isValid()     { return uniqID >= 0; }
    inline void markInvalid() { if( uniqID>=0 ) uniqID = -uniqID-1; }
    inline void markValid()   { if( uniqID<0  ) uniqID = -uniqID-1; }
    inline int  validID()     { return (uniqID<0)?(-uniqID-1):uniqID; }
};


class NTaggedPrim : public NPrim
{
public:
    int tempID;

    inline void untag() { tempID = 0; }
    inline void tag(int t=1) { tempID = t; }
    inline bool isTagged() { return tempID!=0; }
};


// NAUTILUS_NPRIM_INCLUDED
#endif
