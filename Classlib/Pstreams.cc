
/*
 *  Pstreams.cc: reading/writing persistent objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Pstreams.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <fstream.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#define PERSISTENT_VERSION 1

TextPiostream::TextPiostream(ifstream* istr, int version)
: Piostream(Read, version), istr(istr), ostr(0), have_peekname(0)
{
    int fd=istr->rdbuf()->fd();
    struct stat buf;
    if(fstat(fd, &buf) != 0){
	perror("fstat");
	exit(-1);
    }
    len=buf.st_size;
}

TextPiostream::TextPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1)
{
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
	    while(1){
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
    clString name;
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
	while(1){
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
    if(dir == Read){
	int pos=istr->tellg();
	return double(pos)/double(len);
    } else {
	return 0;
    }
}

BinaryPiostream::BinaryPiostream(ifstream* istr, int version)
: Piostream(Read, version), have_peekname(0)
{
    int fd=istr->rdbuf()->fd();
    xdr=scinew XDR;
#ifdef SCI_NOMMAP_IO
    fp=fdopen(istr->rdbuf()->fd(), "r");
    rewind(fp);
    xdrstdio_create(xdr, fp, XDR_DECODE);
#else
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
}

BinaryPiostream::~BinaryPiostream()
{
    cancel_timers();
    if(xdr){
	xdr_destroy(xdr);
	delete xdr;
	if(dir==Read){
#ifdef SCI_NOMMAP_IO
#else
	    if(munmap((caddr_t)addr, len) != 0){
		perror("munmap");
		exit(-1);
	    }
	}
#endif
    }
}

BinaryPiostream::BinaryPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1), have_peekname(0)
{
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
	p=data();
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
