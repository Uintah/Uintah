
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

#include <Persistent.h>
#include <NotFinished.h>
#include <Geometry/Vector.h>
#include <Classlib/String.h>
#include <fstream.h>

#define PERSISTENT_VERSION 1

Persistent::~Persistent()
{
}

Piostream::Piostream(Direction dir, int version)
: dir(dir), version(version), err(0)
{
}

Piostream::~Piostream()
{
}

int Piostream::reading()
{
    return dir==Read;
}

int Piostream::writing()
{
    return dir=Write;
}

int Piostream::error()
{
    return err;
}

void Piostream::io(Vector& v)
{
    begin_cheap_delim();
    if(dir==Read){
	double x,y,z;
	io(x); io(y); io(z);
	v=Vector(x,y,z);
    } else {
	double x=v.x();	
	double y=v.y();
	double z=v.z();
	io(x); io(y); io(z);
    }
    end_cheap_delim();
}

void Piostream::io(Point& p)
{
    begin_cheap_delim();
    if(dir==Read){
	double x,y,z;
	io(x); io(y); io(z);
	p=Point(x,y,z);
    } else {
	double x=p.x();	
	double y=p.y();
	double z=p.z();
	io(x); io(y); io(z);
    }
    end_cheap_delim();
}

Piostream* auto_istream(const clString& filename)
{
    ifstream* inp=new ifstream(filename());
    ifstream& in=(*inp);
    if(!in){
	cerr << "file not found: " << filename << endl;
	return 0;
    }
    char m1, m2, m3, m4;
    // >> Won't work here - it eats white space...
    in.get(m1); in.get(m2); in.get(m3); in.get(m4);
    if(!in || m1 != 'S' || m2 != 'C' || m3 != 'I' || m4 != '\n'){
	cerr << filename << " is not a valid SCI file! (magic=" << m1 << m2 << m3 << m4 << ")\n";
	return 0;
    }
    in.get(m1); in.get(m2); in.get(m3); in.get(m4);
    if(!in){
	cerr << "Error reading file: " << filename << " (while readint type)" << endl;
	return 0;
    }
    int version;
    in >> version;
    if(!in){
	cerr << "Error reading file: " << filename << " (while reading version)" << endl;
	return 0;
    }
    char m;
    do {
	in.get(m);
	if(!in){
	    cerr << "Error reading file: " << filename << " (while reading newline)" << endl;
	    return 0;
	}
    } while(m != '\n');
    if(m1 == 'B' && m2 == 'I' && m3 == 'N'){
	// return new BinaryPiostream(inp, version);
	NOT_FINISHED("Binary Piostream");
	return 0;
    } else if(m1 == 'A' && m2 == 'S' && m3 == 'C'){
	return new TextPiostream(inp, version);
    } else {
	cerr << filename << " is an unknown type!\n";
	return 0;
    }
}

#if 0
BinaryPiostream::BinaryPiostream(ifstream* istr, int version)
: Piostream(Read, version), istr(istr)
{
}

BinaryPiostream::~BinaryPiostream()
{
    if(istr)
	delete istr;
}
#endif
TextPiostream::TextPiostream(ifstream* istr, int version)
: Piostream(Read, version), istr(istr), ostr(0)
{
}

TextPiostream::TextPiostream(const clString& filename, Direction dir)
: Piostream(dir, -1)
{
    if(dir==Read){
	ostr=0;
	istr=0;
	NOT_FINISHED("TextPiostream::TextPiostream...");
    } else {
	istr=0;
	ostr=new ofstream(filename());
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
    if(istr)
	delete istr;
    if(ostr)
	delete ostr;
}

void TextPiostream::begin_class(const clString& classname)
{
    if(err)return;
    if(dir==Read){
	expect('{');
	for(int i=0;i<classname.len();i++){
	    expect(classname(i));
	}
    } else {
	ofstream& out=*ostr;
	out << "{" << classname << " ";
    }
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
	while(1){
	    char c;
	    in >> c;
	    if(!in){
		err=1;
		return;
	    }
	    if('c' == '"')
		break;
	    else
		*p++=c;
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
	err=1;
	return;
    }
    char c;
    in.get(c);
    if(!in){
	err=1;
	return;
    }
    if(c != expected){
	err=1;
	cerr << "Expected: '" << expected << "', got '" << c << "'." << endl;
	char buf[100];
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	return;
    }
}
