/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <teem/air.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>
using namespace std;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <teem/nrrd.h>
#ifdef _WIN32
#include <io.h>
#endif


#define PERSISTENT_VERSION 2


namespace SCIRun {

// BinaryPiostream -- portable
BinaryPiostream::BinaryPiostream(const string& filename, Direction dir,
                                 const int& v)
  : Piostream(dir, -1, filename),
    fp_(0)
{
  if (v == -1) // no version given so use PERSISTENT_VERSION
    version = PERSISTENT_VERSION;
  else
    version = v;

  if (dir==Read)
  {
    fp_ = fopen (filename.c_str(), "rb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for reading\n";
      err = true;
      return;
    }

    // old versions had headers of size 12
    if (version == 1)
    {
      char hdr[12];

      // read header
      if (!fread(hdr, 1, 12, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
    else
    {
      // versions > 1 have size of 16 to
      // account for endianness in header (LIT | BIG)
      char hdr[16];

      // read header
      if (!fread(hdr, 1, 16, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
  }
  else
  {
    fp_=fopen(filename.c_str(), "wb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for writing\n";
      err = true;
      return;
    }

    version=PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nBIN\n%03d\nLIT\n", version);
      else
	sprintf(hdr, "SCI\nBIN\n%03d\nBIG\n", version);

      if (!fwrite(hdr,1,16,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nBIN\n%03d\n", version);
      if (!fwrite(hdr,1,12,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
  }
}


BinaryPiostream::BinaryPiostream(int fd, Direction dir, const int& v)
  : Piostream(dir, -1),
    fp_(0)
{
  if (v == -1) // no version given so use PERSISTENT_VERSION
    version = PERSISTENT_VERSION;
  else
    version = v;

  if (dir==Read)
  {
    fp_ = fdopen (fd, "rb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for reading\n";
      err = true;
      return;
    }

    // old versions had headers of size 12
    if (version == 1)
    {
      char hdr[12];

      // read header
      if (!fread(hdr, 1, 12, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
    else
    {
      // versions > 1 have size of 16 to
      // account for endianness in header (LIT | BIG)
      char hdr[16];

      // read header
      if (!fread(hdr, 1, 16, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
  }
  else
  {
    fp_ = fdopen(fd, "wb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for writing\n";
      err = true;
      return;
    }

    version = PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nBIN\n%03d\nLIT\n", version);
      else
	sprintf(hdr, "SCI\nBIN\n%03d\nBIG\n", version);

      if (!fwrite(hdr,1,16,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nBIN\n%03d\n", version);
      if (!fwrite(hdr,1,12,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
  }
}


BinaryPiostream::~BinaryPiostream()
{
  if (fp_) fclose(fp_);
}


template <class T>
inline void
BinaryPiostream::gen_io(T& data, const char *iotype)
{
  if (err) return;
  if (dir==Read)
  {
    if (!fread(&data, sizeof(short), 1, fp_))
    {
      err = true;
      cerr << "BinaryPiostream error reading " << iotype << "." << endl;
    }
  }
  else
  {
    if (!fwrite(&data, sizeof(short), 1, fp_))
    {
      err = true;
      cerr << "BinaryPiostream error writing " << iotype << "." << endl;
    }
  }
}


void
BinaryPiostream::io(char& data)
{
  gen_io(data, "char");
}


void
BinaryPiostream::io(signed char& data)
{
  gen_io(data, "signed char");
}


void
BinaryPiostream::io(unsigned char& data)
{
  gen_io(data, "unsigned char");
}


void
BinaryPiostream::io(short& data)
{
  gen_io(data, "short");
}


void
BinaryPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}


void
BinaryPiostream::io(int& data)
{
  gen_io(data, "int");
}


void
BinaryPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}


void
BinaryPiostream::io(long& data)
{
  gen_io(data, "long");
}


void
BinaryPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}


void
BinaryPiostream::io(long long& data)
{
  gen_io(data, "long long");
}


void
BinaryPiostream::io(unsigned long long& data)
{
  gen_io(data, "unsigned long long");
}


void
BinaryPiostream::io(double& data)
{
  gen_io(data, "double");
}


void
BinaryPiostream::io(float& data)
{
  gen_io(data, "float");
}


void
BinaryPiostream::io(string& data)
{
  if (err) return;
  unsigned int chars = 0;
  if (dir==Write)
  {
    const char* p=data.c_str();
    chars = static_cast<int>(strlen(p)) + 1;
    io(chars);
    if (!fwrite(p, sizeof(char), chars, fp_)) err = true;
  }
  if (dir==Read)
  {
    // read in size
    io(chars);

    if (version == 1)
    {
      // Some of the property manager's objects write out
      // strings of size 0 followed a character, followed
      // by an unsigned short which is the size of the following
      // string
      if (chars == 0)
      {
	char c;

	// skip character
	fread(&c, sizeof(char), 1, fp_);

	unsigned short s;
	io(s);

	// create buffer which is multiple of 4
	int extra = 4-(s%4);
	if (extra == 4)
	  extra = 0;
	unsigned int buf_size = s+extra;
	char* buf = new char[buf_size];
	
	// read in data plus padding
	if (!fread(buf, sizeof(char), buf_size, fp_))
        {
	  err = true;
	  delete [] buf;
	  return;
	}
	
	// only use actual size of string
	data = "";
	for(unsigned int i=0; i<s; i++)
	  data += buf[i];
	delete [] buf;
      }
      else
      {
	// used to create a buffer which is multiple of 4
	int extra = 4-(chars%4);
	if (extra == 4)
	  extra = 0;
	int buf_size = chars+extra;
	char* buf = new char[buf_size];
	
	// read in data plus padding
	if (!fread(buf, sizeof(char), buf_size, fp_))
        {
	  err = true;
	}
	
	// only use actual size of string
	data = "";
	for(unsigned int i=0; i<chars; i++)
	  data += buf[i];
	
	delete [] buf;
      }
    }
    else
    {
      char* buf = new char[chars];
      fread(buf, sizeof(char), chars, fp_);
      data=string(buf);
      delete[] buf;
    }
  }
}


////
// BinarySwapPiostream -- portable
// Piostream used when endianness of machine and file don't match
BinarySwapPiostream::BinarySwapPiostream(const string& filename, Direction dir,
                                         const int& v)
  : Piostream(dir, -1, filename),
    fp_(0)
{
  if (v == -1) // no version given so use PERSISTENT_VERSION
    version = PERSISTENT_VERSION;
  else
    version = v;
  if (dir==Read)
  {
    fp_ = fopen (filename.c_str(), "rb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for reading\n";
      err = true;
      return;
    }

    if (version == 1)
    {
      char hdr[12];

      // read header
      if (!fread(hdr, 1, 12, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
    else
    {
      // versions > 1 have size of 16 to
      // account for endianness in header (LIT | BIG)
      char hdr[16];

      // read header
      if (!fread(hdr, 1, 16, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
  }
  else
  {
    fp_=fopen(filename.c_str(), "wb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for writing\n";
      err = true;
      return;
    }

    version=PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nBIN\n%03d\nBIG\n", version);
      else
	sprintf(hdr, "SCI\nBIN\n%03d\nLIT\n", version);

      if (!fwrite(hdr,1,16,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nBIN\n%03d\n", version);
      if (!fwrite(hdr,1,12,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
  }
}


BinarySwapPiostream::BinarySwapPiostream(int fd, Direction dir, const int& v)
  : Piostream(dir, -1),
    fp_(0)
{
  if (v == -1) // no version given so use PERSISTENT_VERSION
    version = PERSISTENT_VERSION;
  else
    version = v;
  if (dir==Read)
  {
    fp_ = fdopen (fd, "rb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for reading\n";
      err = true;
      return;
    }

    // old versions had headers of size 12
    if (version == 1)
    {
      char hdr[12];

      // read header
      if (!fread(hdr, 1, 12, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
    else
    {
      // versions > 1 have size of 16 to
      // account for endianness in header (LIT | BIG)
      char hdr[16];

      // read header
      if (!fread(hdr, 1, 16, fp_))
      {
	cerr << "header fread failed\n";
	err = true;
	return;
      }
    }
  }
  else
  {
    fp_=fdopen(fd, "wb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for writing\n";
      err = true;
      return;
    }

    version=PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nBIN\n%03d\nBIG\n", version);
      else
	sprintf(hdr, "SCI\nBIN\n%03d\nLIT\n", version);
      if (!fwrite(hdr,1,16,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nBIN\n%03d\n", version);
      if (!fwrite(hdr,1,12,fp_))
      {
	cerr << "header fwrite failed\n";
	err = true;
	return;
      }
    }
  }
}


BinarySwapPiostream::~BinarySwapPiostream()
{
  if (fp_) fclose(fp_);
}


template <class T>
inline void
BinarySwapPiostream::gen_io(T& data, const char *iotype)
{
  if (err) return;
  if (dir==Read)
  {
    unsigned char tmp[sizeof(data)];
    if (!fread(tmp, sizeof(data), 1, fp_))
    {
      err = true;
      cerr << "BinaryPiostream error reading " << iotype << "." << endl;
      return;
    }
    unsigned char *cdata = reinterpret_cast<unsigned char *>(&data);
    for (unsigned int i = 0; i < sizeof(data); i++)
    {
      cdata[i] = tmp[sizeof(data)-i-1];
    }
  }
  else
  {
    unsigned char tmp[sizeof(data)];
    unsigned char *cdata = reinterpret_cast<unsigned char *>(&data);
    for (unsigned int i = 0; i < sizeof(data); i++)
    {
      tmp[i] = cdata[sizeof(data)-i-1];
    }
    if (!fwrite(tmp, sizeof(data), 1, fp_))
    {
      err = true;
      cerr << "BinaryPiostream error writing " << iotype << "." << endl;
    }
  }
}


void
BinarySwapPiostream::io(char& data)
{
  if (err) return;
  if (dir==Read)
  {
    // no need to swap a single byte
    if (!fread(&data, sizeof(char), 1, fp_)) err = true;
  }
  else
  {
    if (!fwrite(&data, sizeof(char), 1, fp_)) err = true;
  }
}


void
BinarySwapPiostream::io(signed char& data)
{
  if (err) return;
  if (dir==Read)
  {
    // no need to swap a single byte
    if (!fread(&data, sizeof(signed char), 1, fp_)) err = true;
  }
  else
  {
    if (!fwrite(&data, sizeof(signed char), 1, fp_)) err = true;
  }
}


void
BinarySwapPiostream::io(unsigned char& data)
{
  if (err) return;
  if (dir==Read)
  {
    // no need to swap a single byte
    if (!fread(&data, sizeof(unsigned char), 1, fp_)) err = true;
  }
  else
  {
    if (!fwrite(&data, sizeof(unsigned char), 1, fp_)) err = true;
  }
}


void
BinarySwapPiostream::io(short& data)
{
  gen_io(data, "short");
}


void
BinarySwapPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}


void
BinarySwapPiostream::io(int& data)
{
  gen_io(data, "int");
}


void
BinarySwapPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}


void
BinarySwapPiostream::io(long& data)
{
  gen_io(data, "long");
}


void
BinarySwapPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}


void
BinarySwapPiostream::io(long long& data)
{
  gen_io(data, "long long");
}


void
BinarySwapPiostream::io(unsigned long long& data)
{
  gen_io(data, "unsigned long long");
}


void
BinarySwapPiostream::io(double& data)
{
  gen_io(data, "double");
}


void
BinarySwapPiostream::io(float& data)
{
  gen_io(data, "float");
}


void
BinarySwapPiostream::io(string& data)
{
  if (err) return;
  unsigned int chars = 0;
  if (dir==Write)
  {
    const char* p=data.c_str();
    chars = static_cast<int>(strlen(p)) + 1;
    io(chars);
    if (!fwrite(p, sizeof(char), chars, fp_)) err = true;
  }
  if (dir==Read)
  {
    // read in size
    io(chars);

    if (version == 1)
    {
      // Some of the property manager's objects write out
      // strings of size 0 followed a character, followed
      // by an unsigned short which is the size of the following
      // string
      if (chars == 0)
      {
	char c;

	// skip character
	fread(&c, sizeof(char), 1, fp_);

	unsigned short s;
	io(s);

	// create buffer which is multiple of 4
	int extra = 4-(s%4);
	if (extra == 4)
	  extra = 0;
	unsigned int buf_size = s+extra;
	char* buf = new char[buf_size];
	
	// read in data plus padding
	if (!fread(buf, sizeof(char), buf_size, fp_))
        {
	  err = true;
	  delete [] buf;
	  return;
	}
	
	// only use actual size of string
	data = "";
	for(unsigned int i=0; i<s; i++)
	  data += buf[i];
	delete [] buf;
      }
      else
      {
	// create buffer which is multiple of 4
	int extra = 4-(chars%4);
	if (extra == 4)
	  extra = 0;
	unsigned int buf_size = chars+extra;
	char* buf = new char[buf_size];
	
	// read in data plus padding
	if (!fread(buf, sizeof(char), buf_size, fp_))
        {
	  err = true;
	  delete [] buf;
	  return;
	}
	
	// only use actual size of string
	data = "";
	for(unsigned int i=0; i<chars; i++)
	  data += buf[i];
	delete [] buf;
      }
    }
    else
    {
      char* buf = new char[chars];
      fread(buf, sizeof(char), chars, fp_);
      data=string(buf);
      delete[] buf;
    }
  }
}


TextPiostream::TextPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1, filename),
    ownstreams_p_(true)
{
  if (dir==Read)
  {
    ostr=0;
    istr=new ifstream(filename.c_str());
    if (!istr)
    {
      cerr << "Error opening file: " << filename << " for reading\n";
      err = true;
      return;
    }
    char hdr[12];
    istr->read(hdr, 8);
    if (!*istr)
    {
      cerr << "Error reading header of file: " << filename << "\n";
      err = true;
      return;
    }
    int c=8;
    while (*istr && c < 12)
    {
      hdr[c]=istr->get();
      if (hdr[c] == '\n')
	break;
      c++;
    }
    if (!readHeader(filename, hdr, "ASC", version, file_endian))
    {
      cerr << "Error parsing header of file: " << filename << "\n";
      err = true;
      return;
    }
  }
  else
  {
    istr=0;
    ostr=scinew ofstream(filename.c_str());
    ostream& out=*ostr;
    if (!out)
    {
      cerr << "Error opening file: " << filename << " for writing\n";
      err = true;
      return;
    }
    out << "SCI\nASC\n" << PERSISTENT_VERSION << "\n";
    version=PERSISTENT_VERSION;
  }
}


TextPiostream::TextPiostream(istream *strm)
  : Piostream(Read, -1),
    istr(strm),
    ostr(0),
    ownstreams_p_(false)
{
  char hdr[12];
  istr->read(hdr, 8);
  if (!*istr)
  {
    cerr << "Error reading header of istream.\n";
    err = true;
    return;
  }
  int c=8;
  while (*istr && c < 12)
  {
    hdr[c]=istr->get();
    if (hdr[c] == '\n')
      break;
    c++;
  }
  if (!readHeader("istream", hdr, "ASC", version, file_endian))
  {
    cerr << "Error parsing header of istream.\n";
    err = true;
    return;
  }
}


TextPiostream::TextPiostream(ostream *strm)
  : Piostream(Write, -1),
    istr(0),
    ostr(strm),
    ownstreams_p_(false)
{
  ostream& out=*ostr;
  out << "SCI\nASC\n" << PERSISTENT_VERSION << "\n";
  version=PERSISTENT_VERSION;
}


TextPiostream::~TextPiostream()
{
  if (ownstreams_p_)
  {
    if (istr)
      delete istr;
    if (ostr)
      delete ostr;
  }
}


void
TextPiostream::io(int do_quotes, string& str)
{
  if (do_quotes)
  {
    io(str);
  }
  else
  {
    if (dir==Read)
    {
      char buf[1000];
      char* p=buf;
      int n=0;
      istream& in=*istr;
      for(;;)
      {
	char c;
	in.get(c);
	if (!in)
        {
	  cerr << "String input failed\n";
	  char buf[100];
	  in.clear();
	  in.getline(buf, 100);
	  cerr << "Rest of line is: " << buf << endl;
	  err = true;
	  break;
	}
	if (c == ' ')
	  break;
	else
	  *p++=c;
	if (n++ > 998)
        {
	  cerr << "String too long\n";
	  char buf[100];
	  in.clear();
	  in.getline(buf, 100);
	  cerr << "Rest of line is: " << buf << endl;
	  err = true;
	  break;
	}
      }
      *p=0;
      str = string(buf);
    }
    else
    {
      ostream& out=*ostr;
      out << str << " ";
    }
  }
}


string
TextPiostream::peek_class()
{
  expect('{');
  io(0, peekname_);
  have_peekname_ = true;
  return peekname_;
}


int
TextPiostream::begin_class(const string& classname, int current_version)
{
  if (err) return -1;
  int version=current_version;
  string gname;
  if (dir==Write)
  {
    gname=classname;
    ostream& out=*ostr;
    out << '{';
    io(0, gname);
  }
  else if (dir==Read && have_peekname_)
  {
    gname = peekname_;
  }
  else
  {
    expect('{');
    io(0, gname);
  }
  have_peekname_ = false;

  if (dir==Read)
  {
    if (classname != gname)
    {
      err = true;
      cerr << "Expecting class: " << classname << ", got class: " << gname << endl;
      return 0;
    }
  }
  io(version);
  return version;
}


void
TextPiostream::end_class()
{
  if (err) return;

  if (dir==Read)
  {
    expect('}');
    expect('\n');
  }
  else
  {
    ostream& out=*ostr;
    out << "}\n";
  }
}


void
TextPiostream::begin_cheap_delim()
{
  if (err) return;
  if (dir==Read)
  {
    expect('{');
  }
  else
  {
    ostream& out=*ostr;
    out << "{";
  }
}


void
TextPiostream::end_cheap_delim()
{
  if (err) return;
  if (dir==Read)
  {
    expect('}');
  }
  else
  {
    ostream& out=*ostr;
    out << "}";
  }
}


void
TextPiostream::io(bool& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading char\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(char& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading char\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(signed char& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading signed char\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(unsigned char& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading unsigned char\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(short& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading short\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(unsigned short& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading unsigned short\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(int& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading int\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(unsigned int& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading unsigned int\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(long& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading long\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(unsigned long& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading unsigned long\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(long long& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading long long\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(unsigned long long& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      cerr << "Error reading unsigned long long\n";
      char buf[100];
      in.clear();
      in.getline(buf, 100);
      cerr << "Rest of line is: " << buf << endl;
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(double& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;

    if (!in)
    {
      char ibuf[5];
      in.clear();
      in.get(ibuf,4);
      // Set the last character to null for airToLower.
      ibuf[3] = '\0';
      // Make sure the comparison is case insensitive.
      airToLower(ibuf);
      if (strcmp(ibuf,"nan")==0)
      {
        data = (double) AIR_NAN;
      }
      else if (strcmp(ibuf,"inf")==0)
      {
        data = (double) AIR_POS_INF;
      }
      else
      {
        in.clear();
        in.get(&(ibuf[3]),2);
        // Set the last character to null for airToLower.
        ibuf[4] = '\0';
        airToLower(ibuf);
        if (strcmp(ibuf,"-inf")==0)
        {
          data = (double) AIR_NEG_INF;
        }
        else
        {
          char buf[100];
          cerr << "Error reading double\n";
          in.clear();
          in.getline(buf, 100);
          cerr << "Rest of line is: " << ibuf << buf << endl;
          err = true;
          return;
        }
      }
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(float& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    in >> data;
    if (!in)
    {
      char ibuf[5];
      in.clear();
      in.get(ibuf,4);
      // Set the last character to null for airToLower.
      ibuf[3] = '\0';
      // Make sure the comparison is case insensitive.
      airToLower(ibuf);
      if (strcmp(ibuf,"nan")==0)
      {
        data = AIR_NAN;
      }
      else if (strcmp(ibuf,"inf")==0)
      {
        data = AIR_POS_INF;
      }
      else
      {
        in.clear();
        in.get(&(ibuf[3]),2);
        // Set the last character to null for airToLower.
        ibuf[4] = '\0';
        airToLower(ibuf);
        if (strcmp(ibuf,"-inf")==0)
        {
          data = AIR_NEG_INF;
        }  	  	
        else
        {
          char buf[100];
          cerr << "Error reading float\n";
          in.clear();
          in.getline(buf, 100);
          cerr << "Rest of line is: " << ibuf << buf << endl;
          err = true;
          return;
        }
      }
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    out << data << " ";
  }
}


void
TextPiostream::io(string& data)
{
  if (err) return;
  if (dir==Read)
  {
    istream& in=*istr;
    expect('"');
    char buf[1000];
    char* p=buf;
    int n=0;
    for(;;)
    {
      char c;
      in.get(c);
      if (!in)
      {
	cerr << "String input failed\n";
	char buf[100];
	in.clear();
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	err = true;
	return;
      }
      if (c == '"')
	break;
      else
	*p++=c;
      if (n++ > 998)
      {
	cerr << "String too long\n";
	char buf[100];
	in.clear();
	in.getline(buf, 100);
	cerr << "Rest of line is: " << buf << endl;
	err = true;
	break;
      }
    }
    *p=0;
    expect(' ');
    data=string(buf);
  }
  else
  {
    ostream& out=*ostr;
    out << "\"" << data << "\" ";
  }
}


void
TextPiostream::expect(char expected)
{
  if (err) return;
  istream& in=*istr;
  if (!in)
  {
    cerr << "read in expect failed (before read)\n";
    char buf[100];
    in.clear();
    in.getline(buf, 100);
    cerr << "Rest of line is: " << buf << endl;
    in.clear();
    in.getline(buf, 100);
    cerr << "Next line is: " << buf << endl;
    err = true;
    return;
  }
  char c;
  in.get(c);
  if (!in)
  {
    cerr << "read in expect failed (after read)\n";
    char buf[100];
    in.clear();
    in.getline(buf, 100);
    cerr << "Rest of line is: " << buf << endl;
    in.clear();
    in.getline(buf, 100);
    cerr << "Next line is: " << buf << endl;
    err = true;
    return;
  }
  if (c != expected)
  {
    err = true;
    cerr << "Persistent Object Stream: Expected '" << expected << "', got '" << c << "'." << endl;
    char buf[100];
    in.clear();
    in.getline(buf, 100);
    cerr << "Rest of line is: " << buf << endl;
    cerr << "Object is not intact" << endl;
    return;
  }
}


void
TextPiostream::emit_pointer(int& have_data, int& pointer_id)
{
  if (dir==Read)
  {
    istream& in=*istr;
    char c;
    in.get(c);
    if (in && c=='%')
      have_data=0;
    else if (in && c=='@')
      have_data=1;
    else
    {
      cerr << "Error reading pointer...\n";
      err = true;
    }
    in >> pointer_id;
    if (!in)
    {
      cerr << "Error reading pointer id\n";
      err = true;
      return;
    }
    expect(' ');
  }
  else
  {
    ostream& out=*ostr;
    if (have_data)
      out << '@';
    else
      out << '%';
    out << pointer_id << " ";
  }
}


// FastPiostream is a non portable binary output.
FastPiostream::FastPiostream(const string& filename, Direction dir) :
  Piostream(dir, -1, filename),
  fp_(0)
{
  if (dir==Read)
  {
    fp_ = fopen (filename.c_str(), "rb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for reading\n";
      err = true;
      return;
    }
    char hdr[12];
    size_t chars_read = fread(hdr, sizeof(char), 12, fp_);
    if (chars_read != 12)
    {
      cerr << "Error reading header from: " << filename << endl;
      err = true;
      return;
    }
    readHeader(filename, hdr, "FAS", version, file_endian);
  }
  else
  {
    fp_=fopen(filename.c_str(), "wb");
    if (!fp_)
    {
      cerr << "Error opening file: " << filename << " for writing\n";
      err = true;
      return;
    }
    version=PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nFAS\n%03d\nLIT\n", version);
      else
	sprintf(hdr, "SCI\nFAS\n%03d\nBIG\n", version);
      // write the header
      size_t wrote = fwrite(hdr, sizeof(char), 16, fp_);
      if (wrote != 16)
      {
	cerr << "Error writing header to: " << filename << endl;
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nFAS\n%03d\n", version);
      // write the header
      size_t wrote = fwrite(hdr, sizeof(char), 12, fp_);
      if (wrote != 12)
      {
	cerr << "Error writing header to: " << filename << endl;
	err = true;
	return;
      }
    }
  }
}


FastPiostream::FastPiostream(int fd, Direction dir) :
  Piostream(dir, -1),
  fp_(0)
{
  if (dir==Read)
  {
    fp_ = fdopen (fd, "rb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for reading\n";
      err = true;
      return;
    }
    char hdr[12];
    size_t chars_read = fread(hdr, sizeof(char), 12, fp_);
    if (chars_read != 12)
    {
      cerr << "Error reading header from socket: " << fd << endl;
      err = true;
      return;
    }
    readHeader("socket", hdr, "FAS", version, file_endian);
  }
  else
  {
    fp_=fdopen(fd, "wb");
    if (!fp_)
    {
      cerr << "Error opening socket: " << fd << " for writing\n";
      err = true;
      return;
    }
    version=PERSISTENT_VERSION;
    if (version > 1)
    {
      char hdr[16];
      if (airMyEndian == airEndianLittle)
	sprintf(hdr, "SCI\nFAS\n%03d\nLIT\n", version);
      else
	sprintf(hdr, "SCI\nFAS\n%03d\nBIG\n", version);
      // write the header
      size_t wrote = fwrite(hdr, sizeof(char), 16, fp_);
      if (wrote != 16)
      {
	cerr << "Error writing header to: " << fd << endl;
	err = true;
	return;
      }
    }
    else
    {
      char hdr[12];
      sprintf(hdr, "SCI\nFAS\n%03d\n", version);
      // write the header
      size_t wrote = fwrite(hdr, sizeof(char), 12, fp_);
      if (wrote != 12)
      {
	cerr << "Error writing header to socket: " << fd << endl;
	err = true;
	return;
      }
    }
  }
}


FastPiostream::~FastPiostream()
{
  if (fp_) fclose(fp_);
}


template <class T>
inline void
FastPiostream::gen_io(T& data, const char *iotype)
{
  if (err) return;
  size_t expect = 1;
  size_t did = 0;
  if (dir == Read)
  {
    did = fread(&data, sizeof(T), 1, fp_);
    if (expect != did && !feof(fp_))
    {
      err = true;
      cerr << "FastPiostream error reading " << iotype << "." << endl;
    }
  }
  else
  {
    did = fwrite(&data, sizeof(T), 1, fp_);
    if (expect != did)
    {
      err = true;
      cerr << "FastPiostream error writing " << iotype << "." << endl;
    }
  }
}


void
FastPiostream::io(bool& data)
{
  gen_io(data, "bool");
}


void
FastPiostream::io(char& data)
{
  gen_io(data, "char");
}


void
FastPiostream::io(signed char& data)
{
  gen_io(data, "signed char");
}


void
FastPiostream::io(unsigned char& data)
{
  gen_io(data, "unsigned char");
}


void
FastPiostream::io(short& data)
{
  gen_io(data, "short");
}


void
FastPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}


void
FastPiostream::io(int& data)
{
  gen_io(data, "int");
}


void
FastPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}


void
FastPiostream::io(long& data)
{
  gen_io(data, "long");
}


void
FastPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}


void
FastPiostream::io(long long& data)
{
  gen_io(data, "long long");
}


void
FastPiostream::io(unsigned long long& data)
{
  gen_io(data, "unsigned long long");
}


void
FastPiostream::io(double& data)
{
  gen_io(data, "double");
}


void
FastPiostream::io(float& data)
{
  gen_io(data, "float");
}


void
FastPiostream::io(string& data)
{
  if (err) return;
  unsigned int chars = 0;
  if (dir==Write)
  {
    const char* p=data.c_str();
    chars = static_cast<int>(strlen(p)) + 1;
    fwrite(&chars, sizeof(unsigned int), 1, fp_);
    fwrite(p, sizeof(char), chars, fp_);
  }
  if (dir==Read)
  {
    fread(&chars, sizeof(unsigned int), 1, fp_);
    char* buf = new char[chars];
    fread(buf, sizeof(char), chars, fp_);
    data=string(buf);
    delete[] buf;
  }
}


void
FastPiostream::block_io(void *data, size_t s, size_t nmemb)
{
  if (dir == Read)
  {
    const size_t did = fread(data, s, nmemb, fp_);
    if (did != nmemb)
    {
      err = true;
      cerr << "FastPiostream error reading block io." << endl;
    }
  }
  else
  {
    const size_t did = fwrite(data, s, nmemb, fp_);
    if (did != nmemb)
    {
      err = true;
      cerr << "FastPiostream error writing block io." << endl;
    }      
  }
}



GzipPiostream::GzipPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1)
{
  if (dir==Read)
  {
    cerr << "GzipPiostream cannot read\n";
    gzfile_ = 0;
    err = true;
  }
  else
  {
    gzfile_ = gzopen(filename.c_str(), "w");
    char str[100];
    int version = PERSISTENT_VERSION;
    if (version > 1)
    {
      if (airMyEndian == airEndianLittle)
	sprintf(str, "SCI\nGZP\n001\nLIT\n");
      else
	sprintf(str, "SCI\nGZP\n001\nBIG\n");
    }
    else
    {
      sprintf(str, "SCI\nGZP\n001\n");
    }
    gzwrite(gzfile_, str, static_cast<unsigned int>(strlen(str)));
    version=1;
  }
}


GzipPiostream::~GzipPiostream()
{
  if (gzfile_) { gzclose(gzfile_); }
}


template <class T>
inline void
GzipPiostream::gen_io(T& data, const char *iotype)
{
  if (err) return;
  if (dir == Read)
  {
    if (gzread(gzfile_, &data, sizeof(data)) == -1)
    {
      err = true;
      cerr << "gzread of " << iotype << " failed.\n";
    }
  }
  else
  {
    if (!gzwrite(gzfile_, &data, sizeof(data)))
    {
      err = true;
      cerr << "gzwrite of " << iotype << " failed.\n";
    }
  }
}


void
GzipPiostream::io(char& data)
{
  gen_io(data, "char");
}


void
GzipPiostream::io(signed char& data)
{
  gen_io(data, "signed char");
}


void
GzipPiostream::io(unsigned char& data)
{
  gen_io(data, "unsigned char");
}


void
GzipPiostream::io(short& data)
{
  gen_io(data, "short");
}


void
GzipPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}


void
GzipPiostream::io(int& data)
{
  gen_io(data, "int");
}


void
GzipPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}


void
GzipPiostream::io(long& data)
{
  gen_io(data, "long");
}


void
GzipPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}


void
GzipPiostream::io(long long& data)
{
  gen_io(data, "long long");
}


void
GzipPiostream::io(unsigned long long& data)
{
  gen_io(data, "unsigned long long");
}


void
GzipPiostream::io(double& data)
{
  gen_io(data, "double");
}


void
GzipPiostream::io(float& data)
{
  gen_io(data, "float");
}


void
GzipPiostream::io(string& data)
{
  if (err) return;
  if (dir == Read)
  {
    char c = '1';
    while (c != '\0' && !err)
    {
      io(c);
      data+=c;
    }
  }
  else
  {
    if (!gzwrite(gzfile_, (void *)(data.c_str()), data.size()+1))
    {
      err = true;
      cerr << "gzwrite failed\n";
    }
  }
}



GunzipPiostream::GunzipPiostream(const string& filename, Direction dir)
  : Piostream(dir, -1),
    unzipfile_(-1)
{
  unzipfile_ = open(filename.c_str(), O_RDWR, 0666);
  if (unzipfile_ == -1)
  {
    cerr << "Error opening file: " << filename << " for reading\n";
    err = true;
    return;
  }
}


GunzipPiostream::~GunzipPiostream()
{
  if (unzipfile_ > 0) { close(unzipfile_); }
}


template <class T>
inline void
GunzipPiostream::gen_io(T& data, const char *iotype)
{
  if (err) return;
  if (dir == Read)
  {
    if (read(unzipfile_, &data, sizeof(data)) == -1)
    {
      err = true;
      cerr << "unzipread of " << iotype << " failed.\n";
    }
  }
  else
  {
    if (!write(unzipfile_, &data, sizeof(data)))
    {
      err = true;
      cerr << "unzipwrite of " << iotype << " failed.\n";
    }
  }
}


void
GunzipPiostream::io(char& data)
{
  gen_io(data, "char");
}


void
GunzipPiostream::io(signed char& data)
{
  gen_io(data, "signed char");
}


void
GunzipPiostream::io(unsigned char& data)
{
  gen_io(data, "unsigned char");
}


void
GunzipPiostream::io(short& data)
{
  gen_io(data, "short");
}


void
GunzipPiostream::io(unsigned short& data)
{
  gen_io(data, "unsigned short");
}


void
GunzipPiostream::io(int& data)
{
  gen_io(data, "int");
}


void
GunzipPiostream::io(unsigned int& data)
{
  gen_io(data, "unsigned int");
}


void
GunzipPiostream::io(long& data)
{
  gen_io(data, "long");
}


void
GunzipPiostream::io(unsigned long& data)
{
  gen_io(data, "unsigned long");
}


void
GunzipPiostream::io(long long& data)
{
  gen_io(data, "long long");
}


void
GunzipPiostream::io(unsigned long long& data)
{
  gen_io(data, "unsigned long long");
}


void
GunzipPiostream::io(double& data)
{
  gen_io(data, "double");
}


void
GunzipPiostream::io(float& data)
{
  gen_io(data, "float");
}


void
GunzipPiostream::io(string& data)
{
  if (err) return;
  if (dir == Read)
  {
    char c = '1';
    while (c != '\0' && !err)
    {
      io(c);
      data+=c;
    }
  }
  else
  {
    if (!write(unzipfile_, data.c_str(), data.size()+1))
    {
      err = true;
      cerr << "unzipwrite failed\n";
    }
  }
}


} // End namespace SCIRun


