
/*
 *  Pstreams.cc: reading/writing persistent objects
 *
 *  Written by:
 *   Steven G. Parker
 *  Modified by:
 *   Michelle Miller 
 *   Thu Feb 19 17:04:59 MST 1998
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

// KCC stuff
// #include <fstream.h>

#include <sys/types.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <string.h>
#include <io.h>
#else
#include <sys/mman.h>
#endif

#define PERSISTENT_VERSION 1

namespace SCICore {
namespace PersistentSpace {

TextPiostream::TextPiostream(ifstream* istr, int version)
: Piostream(Read, version), istr(istr), ostr(0), have_peekname(0)
{
  // Dd:
  printf(" TextPiostream constructor not working to get KCC to compile\n");

#if 0
    int fd=istr->rdbuf()->fd();
    struct stat buf;
    if(fstat(fd, &buf) != 0){
	perror("fstat");
	exit(-1);
    }
    len=buf.st_size;
#endif
}

TextPiostream::TextPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1)
{
  // Dd:
  printf(" TextPiostream constructor not working to get KCC to compile\n");

#if 0
    if(dir==Read){
	ostr=0;
	istr=0;
	cerr << "TextPiostream CTOR not finished...\n";
    } else {
	istr=0;
	ostr=scinew ofstream(filename());
	ofstream& out=*ostr;
	if(!out){
	    cerr << "Error opening file: " << filename << " for writing\n";
	    err=1;
	    return;
	}
	out << "SCI\nASC\n" << PERSISTENT_VERSION << "\n";
	version=PERSISTENT_VERSION;
    }
#endif
}

TextPiostream::TextPiostream(int fd, Direction dir)
: Piostream(dir, -1)
{
  // Dd:
  printf(" TextPiostream constructor not working to get KCC to compile\n");

#if 0

    if(dir==Read){
        ostr=0;
        istr=0;
        cerr << "TextPiostream CTOR not finished...\n";
    } else {
        istr=0;
        ostr=scinew ofstream(fd);
        ofstream& out=*ostr;
        if(!out){
            cerr << "Error opening file descriptor: " << fd << " for writing\n";
            err=1;
            return;
        }
        out << "SCI\nASC\n" << PERSISTENT_VERSION << "\n";
        version=PERSISTENT_VERSION;
    }
#endif
}

TextPiostream::~TextPiostream()
{
    cancel_timers();
    if(istr)
	delete istr;
    if(ostr)
	delete ostr;
}

void TextPiostream::io(int do_quotes, clString& string)
{
    if(do_quotes){
	io(string);
    } else {
	if(dir==Read){
	    char buf[1000];
	    char* p=buf;
	    int n=0;
	    ifstream& in=*istr;
	    for(;;){
		char c;
		in.get(c);
		if(!in){
		    cerr << "String input failed\n";
		    char buf[100];
		    in.clear();
		    in.getline(buf, 100);
		    cerr << "Rest of line is: " << buf << endl;
		    err=1;
		    break;
		}
		if(c == ' ')
		    break;
		else
		    *p++=c;
		if(n++ > 998){
		    cerr << "String too long\n";
		    char buf[100];
		    in.clear();
		    in.getline(buf, 100);
		    cerr << "Rest of line is: " << buf << endl;
		    err=1;
		    break;
		}
	    }
	    *p=0;
	    string=clString(buf);
	} else {
	    ofstream& out=*ostr;
	    out << string << " ";
	}
    }
}

clString TextPiostream::peek_class()
{
    expect('{');
    io(0, peekname);
    have_peekname=1;
    return peekname;
}

int TextPiostream::begin_class(const clString& classname,
			       int current_version)
{
    if(err)return -1;
    int version=current_version;
    clString gname;
    if(dir==Write){
	gname=classname;
	ostream& out=*ostr;
	out << '{';
	io(0, gname);
    } else if(dir==Read && have_peekname){
	gname=peekname;
    } else {
	expect('{');
	io(0, gname);
    }
    have_peekname=0;

    if(dir==Read){
	if(classname != gname){
	    err=1;
	    cerr << "Expecting class: " << classname << ", got class: " << gname << endl;
	    return 0;
	}
    }
    io(version);
    return version;
}

void TextPiostream::end_class()
{
    if(err)return;

    if(dir==Read){
	expect('}');
	expect('\n');
    } else {
	ofstream& out=*ostr;
	out << "}\n";
    }
}

void TextPiostream::begin_cheap_delim()
{
    if(err)return;
    if(dir==Read){
	expect('{');
    } else {
	ofstream& out=*ostr;
	out << "{";
    }
}

void TextPiostream::end_cheap_delim()
{
    if(err)return;
    if(dir==Read){
	expect('}');
    } else {
	ofstream& out=*ostr;
	out << "}";
    }
}

void TextPiostream::io(char& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading char\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(unsigned char& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading unsigned char\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(short& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading short\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(unsigned short& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading unsigned short\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(int& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading int\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(unsigned int& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading unsigned int\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(long& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading long\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(unsigned long& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading unsigned long\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(double& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading double\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(float& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	in >> data;
	if(!in){
	    cerr << "Error reading float\n";
	    char buf[100];
	    in.clear();
	    in.getline(buf, 100);
	    cerr << "Rest of line is: " << buf << endl;
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	out << data << " ";
    }
}

void TextPiostream::io(clString& data)
{
    if(err)return;
    if(dir==Read){
	ifstream& in=*istr;
	expect('"');
	char buf[1000];
	char* p=buf;
	int n=0;
	for(;;){
	    char c;
	    in.get(c);
	    if(!in){
		cerr << "String input failed\n";
		char buf[100];
		in.clear();
		in.getline(buf, 100);
		cerr << "Rest of line is: " << buf << endl;
		err=1;
		return;
	    }
	    if(c == '"')
		break;
	    else
		*p++=c;
	    if(n++ > 998){
		cerr << "String too long\n";
		char buf[100];
		in.clear();
		in.getline(buf, 100);
		cerr << "Rest of line is: " << buf << endl;
		err=1;
		break;
	    }
	}
	*p=0;
	expect(' ');
	data=clString(buf);
    } else {
	ofstream& out=*ostr;
	out << "\"" << data << "\" ";
    }
}

void TextPiostream::expect(char expected)
{
    if(err)return;
    ifstream& in=*istr;
    if(!in){
	cerr << "read in expect failed (before read)\n";
	char buf[100];
	in.clear();
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	in.clear();
	in.getline(buf, 100);
	cerr << "Next line is: " << buf << endl;
	err=1;
	return;
    }
    char c;
    in.get(c);
    if(!in){
	cerr << "read in expect failed (after read)\n";
	char buf[100];
	in.clear();
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	in.clear();
	in.getline(buf, 100);
	cerr << "Next line is: " << buf << endl;
	err=1;
	return;
    }
    if(c != expected){
	err=1;
	cerr << "Persistent Object Stream: Expected '" << expected << "', got '" << c << "'." << endl;
	char buf[100];
	in.clear();
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	cerr << "Object is not intact" << endl;
	return;
    }
}

void TextPiostream::emit_pointer(int& have_data, int& pointer_id)
{
    if(dir==Read){
	ifstream& in=*istr;
	char c;
	in.get(c);
	if(in && c=='%')
	    have_data=0;
	else if(in && c=='@')
	    have_data=1;
	else {
	    cerr << "Error reading pointer...\n";
	    err=1;
	}
	in >> pointer_id;
	if(!in){
	    cerr << "Error reading pointer id\n";
	    err=1;
	    return;
	}
	expect(' ');
    } else {
	ofstream& out=*ostr;
	if(have_data)
	    out << '@';
	else
	    out << '%';
	out << pointer_id << " ";
    }
}

double TextPiostream::get_percent_done()
{
  // Dd:
  printf(" TextPiostream get_percent_done not working to get KCC to compile\n");

#if 0
    if(dir == Read){
	int pos=istr->tellg();
	return double(pos)/double(len);
    } else {
	return 0;
    }
#endif
return 0; // Dd: delete this.
}

BinaryPiostream::BinaryPiostream(ifstream* istr, int version)
: Piostream(Read, version), have_peekname(0)
{
  // Dd:
  printf(" BinaryPiostream constructor not working to get KCC to compile\n");

#if 0

    int fd=istr->rdbuf()->fd();
    xdr=scinew XDR;
#ifdef SCI_NOMMAP_IO
    mmapped = false;
    fp=fdopen(istr->rdbuf()->fd(), "r");
    rewind(fp);
    xdrstdio_create(xdr, fp, XDR_DECODE);
#else
    mmapped = true;
    struct stat buf;
    if(fstat(fd, &buf) != 0){
	perror("fstat");
	exit(-1);
    }
    len=buf.st_size;
    addr=mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if((long)addr == -1){
	perror("mmap");
	exit(-1);
    }
    xdrmem_create(xdr, (caddr_t)addr, len, XDR_DECODE);
#endif
    char hdr[100];
    if(!xdr_opaque(xdr, (caddr_t)hdr, 12)){
	cerr << "xdr_opaque failed\n";
	err=1;
	return;
    }
#endif
}

BinaryPiostream::BinaryPiostream (int fd, Direction dir)
: Piostream (dir, -1), have_peekname(0)
{
    char hdr[100];
    mmapped = false;
    xdr = scinew XDR;
    if (dir == Read){
   	fp = fdopen (fd, "r");
      	if(!fp){
           cerr << "Error opening file descriptor: " << fd << " for reading\n";
           err=1;
           return;
  	}
	xdrstdio_create (xdr, fp, XDR_DECODE);
    } else {
    	fp = fdopen (fd, "w");
        if(!fp){
       	   cerr << "Error opening file descriptor: " << fd << " for writing\n";
           err=1;
           return;
       	}
   	xdrstdio_create (xdr, fp, XDR_ENCODE);
       	version=PERSISTENT_VERSION;
        sprintf(hdr, "SCI\nBIN\n%03d\n", version);
    }

    // verify header can be translated 
    if(!xdr_opaque(xdr, (caddr_t)hdr, 12)){
        cerr << "xdr_opaque failed\n";
        err=1;
        return;
    }
}

BinaryPiostream::~BinaryPiostream()
{
    cancel_timers();
    if(xdr){
	xdr_destroy(xdr);
	delete xdr;
	if(dir==Read && mmapped){
#ifndef SCI_NOMMAP_IO
	    if(munmap((caddr_t)addr, len) != 0){
		perror("munmap");
		exit(-1);
	    }
#endif
	}
    }
}

BinaryPiostream::BinaryPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1), have_peekname(0)
{
    mmapped = false;
    if(dir==Read){
	fp=0;
	xdr=0;
	cerr << "BinaryPiostream CTOR not finished...\n";
    } else {
	fp=fopen(filename(), "w");
	if(!fp){
	    cerr << "Error opening file: " << filename << " for writing\n";
	    err=1;
	    return;
	}
	xdr=scinew XDR;
	xdrstdio_create(xdr, fp, XDR_ENCODE);
	char hdr[100];
	version=PERSISTENT_VERSION;
	sprintf(hdr, "SCI\nBIN\n%03d\n", version);
	if(!xdr_opaque(xdr, (caddr_t)hdr, 12)){
	    cerr << "xdr_opaque failed\n";
	    err=1;
	    return;
	}
    }
}

clString BinaryPiostream::peek_class()
{
    have_peekname=1;
    io(peekname);
    return peekname;
}

int BinaryPiostream::begin_class(const clString& classname,
			       int current_version)
{
    if(err)return -1;
    int version=current_version;
    clString gname;
    if(dir==Write){
	gname=classname;
	io(gname);
    } else if(dir==Read && have_peekname){
	gname=peekname;
    } else {
	io(gname);
    }
    have_peekname=0;

    if(dir==Read){
	if(classname != gname){
	    err=1;
	    cerr << "Expecting class: " << classname << ", got class: " << gname << endl;
	    return 0;
	}
    }
    io(version);
    return version;
}

void BinaryPiostream::end_class()
{
    // No-op
}

void BinaryPiostream::begin_cheap_delim()
{
    // No-op
}

void BinaryPiostream::end_cheap_delim()
{
    // No-op
}

void BinaryPiostream::io(char& data)
{
    if(err)return;
    if(!xdr_char(xdr, &data)){
	err=1;
	cerr << "xdr_char failed\n";
    }
}

void BinaryPiostream::io(unsigned char& data)
{
    if(err)return;
    if(!xdr_u_char(xdr, &data)){
	err=1;
	cerr << "xdr_u_char failed\n";
    }
}

void BinaryPiostream::io(short& data)
{
    if(err)return;
    if(!xdr_short(xdr, &data)){
	err=1;
	cerr << "xdr_short failed\n";
    }
}

void BinaryPiostream::io(unsigned short& data)
{
    if(err)return;
    if(!xdr_u_short(xdr, &data)){
	err=1;
	cerr << "xdr_u_short failed\n";
    }
}

void BinaryPiostream::io(int& data)
{
    if(err)return;
    if(!xdr_int(xdr, &data)){
	err=1;
	cerr << "xdr_int failed\n";
    }
}

void BinaryPiostream::io(unsigned int& data)
{
    if(err)return;
    if(!xdr_u_int(xdr, &data)){
	err=1;
	cerr << "xdr_u_int failed\n";
    }
}

void BinaryPiostream::io(long& data)
{
    if(err)return;
    if(!xdr_long(xdr, &data)){
	err=1;
	cerr << "xdr_long failed\n";
    }
}

void BinaryPiostream::io(unsigned long& data)
{
    if(err)return;
    if(!xdr_u_long(xdr, &data)){
	err=1;
	cerr << "xdr_u_long failed\n";
    }
}

void BinaryPiostream::io(double& data)
{
    if(err)return;
    if(!xdr_double(xdr, &data)){
	err=1;
	cerr << "xdr_double failed\n";
    }
}

void BinaryPiostream::io(float& data)
{
    if(err)return;
    if(!xdr_float(xdr, &data)){
	err=1;
	cerr << "xdr_float failed\n";
    }
}

void BinaryPiostream::io(clString& data)
{
    if(err)return;
    char* p=0;
    if(dir==Write)
	p=const_cast<char *>(data());
    if(!xdr_wrapstring(xdr, &p)){
	err=1;
	cerr << "xdr_wrapstring failed\n";
    }
    if(dir==Read){
	data=clString(p);
	free(p);
    }
}

void BinaryPiostream::emit_pointer(int& have_data, int& pointer_id)
{
    if(!xdr_int(xdr, &have_data)){
	err=1;
	cerr << "xdr_int failed\n";
	return;
    }
    if(!xdr_int(xdr, &pointer_id)){
	err=1;
	cerr << "xdr_int failed\n";
	return;
    }
}

double BinaryPiostream::get_percent_done()
{
    if(dir == Read){
	int pos=xdr_getpos(xdr);
	return double(pos)/double(len);
    } else {
	return 0;
    }
}

GzipPiostream::GzipPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1)
{
    if(dir==Read){
	cerr << "GzipPiostream CTOR not finished...\n";
	gzfile=0;
    } else {
	gzfile=gzopen(filename(), "w");
	char str[100];
	sprintf(str, "SCI\nGZP\n1\n");
	gzwrite(gzfile, str, strlen(str));
	version=1;
    }
}

GzipPiostream::GzipPiostream(char* name, int version)
: Piostream(Read, version), have_peekname(0)
{
    gzfile=gzopen(name, "r");
    char str[10];
    gzread(gzfile, str, 10);
    char hdr[13];
    sprintf(hdr, "SCI\nGZP\n%d\n", 1);
    if (strncmp(str, hdr, 10)) {
	gzclose(gzfile);
	gzfile=0;
	return;
    }
}

GzipPiostream::~GzipPiostream()
{
    cancel_timers();
    gzclose(gzfile);
}

clString GzipPiostream::peek_class()
{
    have_peekname=1;
    io(peekname);
    return peekname;
}

int GzipPiostream::begin_class(const clString& classname,
			       int current_version)
{
    if(err)return -1;
    int version=current_version;
    clString gname;
    if(dir==Write){
	gname=classname;
	io(gname);
    } else if(dir==Read && have_peekname){
	gname=peekname;
    } else {
	io(gname);
    }
    have_peekname=0;

    if(dir==Read){
	if(classname != gname){
	    err=1;
	    cerr << "Expecting class: " << classname << ", got class: " << gname << endl;
	    return 0;
	}
    }
    io(version);
    return version;
}

void GzipPiostream::end_class()
{
    // No-op
}

void GzipPiostream::begin_cheap_delim()
{
    // No-op
}

void GzipPiostream::end_cheap_delim()
{
    // No-op
}

void GzipPiostream::io(char& data)
{
    int sz=sizeof(char);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(unsigned char& data)
{
    int sz=sizeof(unsigned char);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(short& data)
{
    int sz=sizeof(short);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(unsigned short& data)
{
    int sz=sizeof(unsigned short);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(int& data)
{
    int sz=sizeof(int);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(unsigned int& data)
{
    int sz=sizeof(unsigned int);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(long& data)
{
    int sz=sizeof(long);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(unsigned long& data)
{
    int sz=sizeof(unsigned long);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(double& data)
{
    int sz=sizeof(double);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(float& data)
{
    int sz=sizeof(float);
    if(err)return;
    if(dir == Read) {
	if (gzread(gzfile, &data, sz) == -1) {
	    err=1;
	    cerr << "gzread failed\n";
	}
    } else {
	if (!gzwrite(gzfile, &data, sz)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::io(clString& data)
{
    if(err)return;
    if(dir == Read) {
	char c='1';
	while (c != '\0' && !err) {
	    io(c);
	    data+=c;
	}
    } else {
	int sz=strlen(data());
	if (!gzwrite(gzfile, const_cast<char *>(data()), sz+1)) {
	    err=1;
	    cerr << "gzwrite failed\n";
	}
    }
}

void GzipPiostream::emit_pointer(int& have_data, int& pointer_id)
{
    io(have_data);
    io(pointer_id);
}

double GzipPiostream::get_percent_done()
{
    return 0;
}

GunzipPiostream::GunzipPiostream(ifstream* istr, int version)
: Piostream(Read, version), have_peekname(0)
{
  // Dd:
  printf(" GunzipPiostream constructor not working to get KCC to compile\n");

#if 0
    unzipfile=istr->rdbuf()->fd();
    struct stat buf;
    if(fstat(unzipfile, &buf) != 0){
	perror("fstat");
	exit(-1);
    }
    len=buf.st_size;
#endif
}

GunzipPiostream::~GunzipPiostream()
{
    cancel_timers();
    close(unzipfile);
}

clString GunzipPiostream::peek_class()
{
    have_peekname=1;
    io(peekname);
    return peekname;
}

int GunzipPiostream::begin_class(const clString& classname,
			       int current_version)
{
    if(err)return -1;
    int version=current_version;
    clString gname;
    if(dir==Write){
	gname=classname;
	io(gname);
    } else if(dir==Read && have_peekname){
	gname=peekname;
    } else {
	io(gname);
    }
    have_peekname=0;

    if(dir==Read){
	if(classname != gname){
	    err=1;
	    cerr << "Expecting class: " << classname << ", got class: " << gname << endl;
	    return 0;
	}
    }
    io(version);
    return version;
}

void GunzipPiostream::end_class()
{
    // No-op
}

void GunzipPiostream::begin_cheap_delim()
{
    // No-op
}

void GunzipPiostream::end_cheap_delim()
{
    // No-op
}

void GunzipPiostream::io(char& data)
{
    int sz=sizeof(char);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(unsigned char& data)
{
    int sz=sizeof(unsigned char);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(short& data)
{
    int sz=sizeof(short);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(unsigned short& data)
{
    int sz=sizeof(unsigned short);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(int& data)
{
    int sz=sizeof(int);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(unsigned int& data)
{
    int sz=sizeof(unsigned int);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(long& data)
{
    int sz=sizeof(long);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(unsigned long& data)
{
    int sz=sizeof(unsigned long);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(double& data)
{
    int sz=sizeof(double);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(float& data)
{
    int sz=sizeof(float);
    if(err)return;
    if(dir == Read) {
	if (read(unzipfile, &data, sz) == -1) {
	    err=1;
	    cerr << "unzipread failed\n";
	}
    } else {
	if (!write(unzipfile, &data, sz)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::io(clString& data)
{
    if(err)return;
    if(dir == Read) {
	char c='1';
	while (c != '\0' && !err) {
	    io(c);
	    data+=c;
	}
    } else {
	int sz=strlen(data());
	if (!write(unzipfile, data(), sz+1)) {
	    err=1;
	    cerr << "unzipwrite failed\n";
	}
    }
}

void GunzipPiostream::emit_pointer(int& have_data, int& pointer_id)
{
    io(have_data);
    io(pointer_id);
}

double GunzipPiostream::get_percent_done()
{
    return 0;
}

} // End namespace PersistentSpace
} // End namespace SCICore

//
// $log$
//
