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

#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using namespace std;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/mman.h>
#endif

// According to (at least) one man page, xdr_hyper() is the same as
// xdr_longlong_t().  AIX does not have xdr_longlong_t, hence...
#if defined(_AIX) && defined(_LONG_LONG)
#  define xdr_longlong_t xdr_hyper
#endif

#ifdef __digital__
typedef longlong_t __int64_t;
#endif

#define PERSISTENT_VERSION 1

namespace SCIRun {

TextPiostream::TextPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1, filename)
{
  if(dir==Read){
    ostr=0;
    istr=new ifstream(filename.c_str());
    if(!istr){
      cerr << "Error opening file: " << filename << " for reading\n";
      err=1;
      return;
    }
    char hdr[12];
    istr->read(hdr, 8);
    if(!*istr){
      cerr << "Error reading header of file: " << filename << "\n";
      err=1;
      return;
    }
    int c=8;
    while (*istr && c < 12){
      hdr[c]=istr->get();
      if(hdr[c] == '\n')
	break;
      c++;
    }
    if(!readHeader(filename, hdr, "ASC", version)){
      cerr << "Error parsing header of file: " << filename << "\n";
      err=1;
      return;
    }
  } else {
    istr=0;
    ostr=scinew ofstream(filename.c_str());
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

void TextPiostream::io(int do_quotes, string& str)
{
  if(do_quotes){
    io(str);
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
      str = string(buf);
    } else {
      ofstream& out=*ostr;
      out << str << " ";
    }
  }
}

string TextPiostream::peek_class()
{
  expect('{');
  io(0, peekname);
  have_peekname=1;
  return peekname;
}

int TextPiostream::begin_class(const string& classname,
			       int current_version)
{
  if(err)return -1;
  int version=current_version;
  string gname;
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

void TextPiostream::io(bool& data)
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

void TextPiostream::io(long long& data)
{
  if(err)return;
  if(dir==Read){
    ifstream& in=*istr;
    in >> data;
    if(!in){
      cerr << "Error reading long long\n";
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

void TextPiostream::io(string& data)
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
    data=string(buf);
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

// FastPiostream is a non portable binary output.
FastPiostream::~FastPiostream()
{
  if (fp_) fclose(fp_);
}

FastPiostream::FastPiostream(const string& filename, Direction dir) : 
  Piostream(dir, -1, filename), 
  fp_(0),
  have_peekname_(0)
{
  if(dir==Read){
    fp_ = fopen (filename.c_str(), "r");
    if(!fp_){
      cerr << "Error opening file: " << filename << " for reading\n";
      err=1;
      return;
    }
    char hdr[12];
    size_t chars_read = fread(hdr, sizeof(char), 12, fp_);
    if (chars_read != 12) {
      cerr << "Error reading header from: " << filename << endl;
      err=1;
      return; 
    }
    readHeader(filename, hdr, "FAS", version);
  } else {
    fp_=fopen(filename.c_str(), "w");
    if(!fp_){
      cerr << "Error opening file: " << filename << " for writing\n";
      err=1;
      return;
    }
    char hdr[12];
    version=PERSISTENT_VERSION;
    sprintf(hdr, "SCI\nFAS\n%03d\n", version);
    // write the header
    size_t wrote = fwrite(hdr, sizeof(char), 12, fp_);
    if (wrote != 12) {
      cerr << "Error writing header to: " << filename << endl;
      err=1;
      return; 
    }
  }
}

FastPiostream::FastPiostream(int fd, Direction dir) : 
  Piostream(dir, -1),
  fp_(0),
  have_peekname_(0)
{
  if(dir==Read){
    fp_ = fdopen (fd, "r");
    if(!fp_){
      cerr << "Error opening socket: " << fd << " for reading\n";
      err=1;
      return;
    }
    char hdr[12];
    size_t chars_read = fread(hdr, sizeof(char), 12, fp_);
    if (chars_read != 12) {
      cerr << "Error reading header from socket: " << fd << endl;
      err=1;
      return; 
    }
    readHeader("socket", hdr, "FAS", version);
  } else {
    fp_=fdopen(fd, "w");
    if(!fp_){
      cerr << "Error opening socket: " << fd << " for writing\n";
      err=1;
      return;
    }
    char hdr[12];
    version=PERSISTENT_VERSION;
    sprintf(hdr, "SCI\nFAS\n%03d\n", version);
    // write the header
    size_t wrote = fwrite(hdr, sizeof(char), 12, fp_);
    if (wrote != 12) {
      cerr << "Error writing header to socket: " << fd << endl;
      err=1;
      return; 
    }
  }
}

string FastPiostream::peek_class()
{
  have_peekname_=1;
  io(peekname_);
  return peekname_;
}

int FastPiostream::begin_class(const string& classname,
			       int current_version)
{
  if(err)return -1;
  int version=current_version;
  string gname;
  if(dir==Write){
    gname=classname;
    io(gname);
  } else if(dir==Read && have_peekname_){
    gname=peekname_;
  } else {
    io(gname);
  }
  have_peekname_=0;

  if(dir==Read){
    if(classname != gname){
      err=1;
      cerr << "Expecting class: " << classname << ", got class: " 
	   << gname << endl;
      return 0;
    }
  }
  io(version);
  return version;
}

void FastPiostream::error(const string &iotype) 
{
  err = 1;
  if (dir == Read) {
    cerr << "FastPiostream error reading ";
  } else {
    cerr << "FastPiostream error writing ";
  }
  cerr << iotype << endl;
}

void FastPiostream::end_class()
{
  // No-op
}

void FastPiostream::begin_cheap_delim()
{
  // No-op
}

void FastPiostream::end_cheap_delim()
{
  // No-op
}
template <class T>
void FastPiostream::gen_io(T& data, const string &iotype) 
{
  if (err) return;
  size_t expect = 1;
  size_t did = 0;
  if (dir == Read) {
    did = fread(&data, sizeof(T), 1, fp_);
    if (expect != did && !feof(fp_)) { error(iotype); }
  } else {
    did = fwrite(&data, sizeof(T), 1, fp_);
    if (expect != did) { error(iotype); }
  }
}
void FastPiostream::io(bool& data)
{
  gen_io(data, "bool");
}

void FastPiostream::io(char& data)
{
  gen_io(data, "char");
}

void FastPiostream::io(unsigned char& data)
{
  gen_io(data, "unsigned char");
}

void FastPiostream::io(short& data)
{
  gen_io(data, "short");
}

void FastPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}

void FastPiostream::io(int& data)
{
  gen_io(data, "int");
}

void FastPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}

void FastPiostream::io(long& data)
{
  gen_io(data, "long");
}

void FastPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}

void FastPiostream::io(long long& data)
{
  gen_io(data, "long long");
}

void FastPiostream::io(double& data)
{
  gen_io(data, "double");
}

void FastPiostream::io(float& data)
{
  gen_io(data, "float");
}

void FastPiostream::io(string& data)
{
  if(err)return;
  unsigned int chars = 0;
  if(dir==Write) {
    const char* p=data.c_str();
    chars = static_cast<int>(strlen(p)) + 1;
    fwrite(&chars, sizeof(unsigned int), 1, fp_);
    fwrite(p, sizeof(char), chars, fp_);
  }
  if(dir==Read){
    fread(&chars, sizeof(unsigned int), 1, fp_);
    char* buf = new char[chars];
    fread(buf, sizeof(char), chars, fp_);
    data=string(buf);
    delete[] buf;
  }
}

void FastPiostream::block_io(void *data, size_t s, size_t nmemb) 
{
  size_t did = 0;
  if (dir == Read) {
    did = fread(data, s, nmemb, fp_); 
  } else {
    did = fwrite(data, s, nmemb, fp_); 
  }
  if (did != nmemb) {
    error("block io");
  }
}
void FastPiostream::emit_pointer(int& have_data, int& pointer_id)
{
  if (err) return;
  io(have_data);
  if (err) return;
  io(pointer_id);
}

// BinaryPiostream -- portable
BinaryPiostream::~BinaryPiostream()
{
#if defined(__APPLE__) && (__GNUC__ == 3) && (__GNUC_MINOR__  <= 3)
  if (xdr)
    if ((xdr)->x_ops)
      if ((xdr)->x_ops->x_destroy)
	(*(xdr)->x_ops->x_destroy)();
  delete xdr;
#else
  if(xdr){
    xdr_destroy(xdr);
    delete xdr;
  }
#endif
}

BinaryPiostream::BinaryPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1, filename), have_peekname(0)
{
  mmapped = false;
  if(dir==Read){
    fp = fopen (filename.c_str(), "r");
    if(!fp){
      cerr << "Error opening file: " << filename << " for reading\n";
      err=1;
      xdr=0;
      return;
    }
    xdr=scinew XDR;
    xdrstdio_create (xdr, fp, XDR_DECODE);

    char hdr[12];
    if(!xdr_opaque(xdr, (caddr_t)hdr, 12)){
      cerr << "xdr_opaque failed\n";
      err=1;
      return;
    }
    readHeader(filename, hdr, "BIN", version);
  } else {
    fp=fopen(filename.c_str(), "w");
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

BinaryPiostream::BinaryPiostream(int fd, Direction dir)
  : Piostream(dir, -1), have_peekname(0)
{
  mmapped = false;
  if(dir==Read){
    fp = fdopen (fd, "r");
    if(!fp){
      cerr << "Error opening socket: " << fd << " for reading\n";
      err=1;
      xdr=0;
      return;
    }
    xdrstdio_create (xdr, fp, XDR_DECODE);

    char hdr[12];
    if(!xdr_opaque(xdr, (caddr_t)hdr, 12)){
      cerr << "xdr_opaque failed\n";
      err=1;
      return;
    }
    version=1;
  } else {
    fp=fdopen(fd, "w");
    if(!fp){
      cerr << "Error opening socket: " << fd << " for writing\n";
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

string BinaryPiostream::peek_class()
{
  have_peekname=1;
  io(peekname);
  return peekname;
}

int BinaryPiostream::begin_class(const string& classname,
				 int current_version)
{
  if(err)return -1;
  int version=current_version;
  string gname;
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

void BinaryPiostream::io(bool& data)
{
  unsigned char tmp = data;
  io(tmp);
  if (dir == Read)
  {
    data = tmp;
  }
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

void BinaryPiostream::io(long long& data)
{
  if(err)return;
#if !defined(__APPLE__) && !defined(__osf__)
  if(!xdr_longlong_t(xdr, (int64_t*)(&data))){
    err=1;
    cerr << "xdr_longlong_t failed\n";
  }
#else
  cerr << "xdr_longlong_t is not implemented on Apple\n";
  err = 1;
#endif
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

void BinaryPiostream::io(string& data)
{
  if(err)return;
  char* p=0;
  if(dir==Write) {
    p = ccast_unsafe(data);
  }
  if(!xdr_wrapstring(xdr, &p)){
    err=1;
    cerr << "xdr_wrapstring failed\n";
  }
  if(dir==Read){
    data=string(p);
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

GzipPiostream::GzipPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1)
{
  if(dir==Read){
    cerr << "GzipPiostream cannot read\n";
    gzfile=0;
  } else {
    gzfile=gzopen(filename.c_str(), "w");
    char str[100];
    sprintf(str, "SCI\nGZP\n001\n");
    gzwrite(gzfile, str, static_cast<unsigned int>(strlen(str)));
    version=1;
  }
}

GzipPiostream::~GzipPiostream()
{
  gzclose(gzfile);
}

string GzipPiostream::peek_class()
{
  have_peekname=1;
  io(peekname);
  return peekname;
}

int GzipPiostream::begin_class(const string& classname,
			       int current_version)
{
  if(err)return -1;
  int version=current_version;
  string gname;
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

void GzipPiostream::io(bool& data)
{
  unsigned char tmp = data;
  io(tmp);
  if (dir == Read)
  {
    data = tmp;
  }
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

void GzipPiostream::io(long long& data)
{
  int sz=sizeof(long long);
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

void GzipPiostream::io(string& data)
{
  if(err)return;
  if(dir == Read) {
    char c='1';
    while (c != '\0' && !err) {
      io(c);
      data+=c;
    }
  } else {
    int sz=static_cast<int>(data.size());
    if (!gzwrite(gzfile, (void *)(data.c_str()), sz+1)) {
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

GunzipPiostream::GunzipPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1), have_peekname(0)
{
  unzipfile=open(filename.c_str(), O_RDWR, 0666);
  if(unzipfile == -1){
    cerr << "Error opening file: " << filename << " for reading\n";
    err=1;
    return;
  }
}

GunzipPiostream::~GunzipPiostream()
{
  close(unzipfile);
}

string GunzipPiostream::peek_class()
{
  have_peekname=1;
  io(peekname);
  return peekname;
}

int GunzipPiostream::begin_class(const string& classname,
				 int current_version)
{
  if(err)return -1;
  int version=current_version;
  string gname;
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

void GunzipPiostream::io(bool& data)
{
  unsigned char tmp = data;
  io(tmp);
  if (dir == Read)
  {
    data = tmp;
  }
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

void GunzipPiostream::io(long long& data)
{
  int sz=sizeof(long long);
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

void GunzipPiostream::io(string& data)
{
  if(err)return;
  if(dir == Read) {
    char c='1';
    while (c != '\0' && !err) {
      io(c);
      data+=c;
    }
  } else {
    int sz=static_cast<int>(data.size());
    if (!write(unzipfile, data.c_str(), sz+1)) {
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

} // End namespace SCIRun


