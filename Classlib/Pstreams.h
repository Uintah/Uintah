
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

#include <Classlib/Persistent.h>
#include <Classlib/String.h>
#include <stdio.h>
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <zlib.h>

class ifstream;
class ofstream;

class BinaryPiostream : public Piostream {
    FILE* fp;
    void* addr;
    int len;
    XDR* xdr;
    bool mmapped;
    virtual void emit_pointer(int&, int&);
    int have_peekname;
    clString peekname;
    virtual double get_percent_done();
public:
    BinaryPiostream(const clString& filename, Direction dir);
    BinaryPiostream(int fd, Direction dir);
    BinaryPiostream(ifstream*, int);

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

class TextPiostream : public Piostream {
    ifstream* istr;
    ofstream* ostr;
    int len;
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
    virtual double get_percent_done();
public:
    TextPiostream(const clString& filename, Direction dir);
    TextPiostream(int fd, Direction dir);
    TextPiostream(ifstream*, int);
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

class GzipPiostream : public Piostream {
    gzFile gzfile;
    int len;
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
    virtual double get_percent_done();
public:
    GzipPiostream(const clString& filename, Direction dir);
    GzipPiostream(char *name, int version);
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

class GunzipPiostream : public Piostream {
    int unzipfile;	// file descriptor
    int len;
    int have_peekname;
    clString peekname;
    void expect(char);
    virtual void emit_pointer(int&, int&);
    virtual double get_percent_done();
public:
    GunzipPiostream(ifstream* istream, int version);
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

#endif
