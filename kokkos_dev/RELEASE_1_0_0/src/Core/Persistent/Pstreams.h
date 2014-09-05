/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/share/share.h>

#include <Core/Persistent/Persistent.h>
#include <Core/Containers/String.h>
#include <stdio.h>
#ifdef _WIN32
#define ZEXPORT __stdcall
#define ZEXTERN extern "C"
#endif
#include <zlib.h>
#include <rpc/types.h>
#include <rpc/xdr.h>

#include <iosfwd>

namespace SCIRun {


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

    virtual void io(bool&);
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

    virtual void io(bool&);
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

    virtual void io(bool&);
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

    virtual void io(bool&);
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

} // End namespace SCIRun


#endif
