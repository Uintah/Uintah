
/*
 *  GeometryPort.h: Handle to the Geometry Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_GeometryPort_h
#define SCI_project_GeometryPort_h 1

#include <Persistent.h>
#include <Port.h>
#include <Geometry/Vector.h>
#include <Geometry/Point.h>

class GeometryIPort : public IPort {
public:
    enum Protocol {
	Atomic=0x01,
    };

protected:
    friend class GeometryOPort;
public:
    GeometryIPort(Module*, const clString& name, int protocol);
    virtual ~GeometryIPort();

    virtual void reset();
    virtual void finish();
};

class GeometryOPort : public OPort {
    GeometryIPort* in;
public:
    GeometryOPort(Module*, const clString& name, int protocol);
    virtual ~GeometryOPort();

    virtual void reset();
    virtual void finish();
};

#endif /* SCI_project_GeometryPort_h */
