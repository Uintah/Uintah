
/*
 *  Pstream.h: reading/writing persistent objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Pstream_h
#define SCI_project_Pstream_h 1

#include <SCICore/share/share.h>

#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Containers/String.h>
#include <stdio.h>
#ifdef _WIN32
#define ZEXPORT __stdcall
#define ZEXTERN extern "C"
#endif
#include <zlib.h>
#include <rpc/types.h>
#include <rpc/xdr.h>

#include <iosfwd>

namespace SCICore {
namespace PersistentSpace {

using SCICore::Containers::clString;

class SCICORESHARE BinaryPiostream : public Piostream {
    FILE* fp;
    void* addr;
    XDR* xdr;
    bool mmapped;
    virtual void emit_pointer(int&, int&);
    int have_peekname;
    clString peekname;
public:
    BinaryPiostream(const clString& filename, Direction dir);
    BinaryPiostream(int fd, Direction dir);

    virtual ~BinaryPiostream();
    virtual clString peek_class();
    virtual int begin_class(const clString& name, int);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
};

class SCICORESHARE TextPiostream : public Piostream {
    std::ifstream* istr;
    std::ofstream* ostr;
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
public:
    TextPiostream(const clString& filename, Direction dir);
    virtual ~TextPiostream();
    virtual clString peek_class();
    virtual int begin_class(const clString& name, int);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
    void io(int, clString& string);
};

class SCICORESHARE GzipPiostream : public Piostream {
    gzFile gzfile;
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
public:
    GzipPiostream(const clString& filename, Direction dir);
    virtual ~GzipPiostream();
    virtual clString peek_class();
    virtual int begin_class(const clString& name, int);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
    void io(int, clString& string);
    inline int fileOpen() { return (gzfile!=0); }
};

class SCICORESHARE GunzipPiostream : public Piostream {
    int unzipfile;	// file descriptor
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
public:
    GunzipPiostream(const clString& filename, Direction dir);
    virtual ~GunzipPiostream();
    virtual clString peek_class();
    virtual int begin_class(const clString& name, int);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
    void io(int, clString& string);
    inline int fileOpen() { return (unzipfile!=0); }
};

} // End namespace PersistentSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/10/07 02:08:02  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:39:41  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:10  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:22  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:30  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
