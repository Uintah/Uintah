/*----------------------------------------------------------------------
CLASS
    ParticleGridReaderPort

    
OVERVIEW TEXT
    


KEYWORDS

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 8, 1999
----------------------------------------------------------------------*/


#ifndef SCI_project_ParticleGridReaderPort_h
#define SCI_project_ParticleGridReaderPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ParticleGridReader.h>

typedef Mailbox<SimplePortComm<ParticleGridReader>*> _cfront_bug_ParticleGridReader_;
typedef SimpleIPort<ParticleGridReaderHandle> ParticleGridReaderIPort;
typedef SimpleOPort<ParticleGridReaderHandle> ParticleGridReaderOPort;

#endif
