
/*
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Persistent_h
#define SCI_project_Persistent_h 1

class clString;
class Pistream;
class Postream;

class Persistent {
public:
    virtual ~Persistent();
    virtual void read(Postream&)=0;
    virtual void write(Pistream&)=0;
    virtual void get_info(clString& name, int& version);
};

#endif
