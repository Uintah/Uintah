
/*
 *  TYPEReader.h: TYPE Reader class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_TYPEReader_h
#define SCI_project_TYPEReader_h 1

#include <Module.h>
#include <TCLvar.h>
#include <TYPEPort.h>
#include <TYPE.h>

class TYPEReader : public Module {
    TYPEOPort* outport;
    TCLstring filename;
    TYPEHandle handle;
public:
    TYPEReader(const clString& id);
    TYPEReader(const TYPEReader&, int deep=0);
    virtual ~TYPEReader();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
