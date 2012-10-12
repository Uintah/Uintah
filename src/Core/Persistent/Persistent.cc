/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 */

#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/StringUtil.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <sci_defs/teem_defs.h>

#ifdef HAVE_TEEM
#  include <teem/nrrd.h>
#else
#  include <Core/Util/Endian.h>
#endif

using namespace std;

#define DEBUG 0

namespace SCIRun {

static Piostream::MapStringPersistentTypeID* table = 0;  
const int Piostream::PERSISTENT_VERSION = 2;

#ifdef __APPLE__
  // On the Mac, this comes from Core/Util/DynamicLoader.cc because
  // the constructor will actually fire from there.  When it is declared
  // in this file, it does not "construct" and thus causes seg faults.
  // (Yes, this is a hack.  Perhaps this problem will go away in later
  // OSX releases, but probably not as it has something to do with the
  // Mac philosophy on when to load dynamic libraries.)
  extern Mutex persistentTypeIDMutex;
#else
  Mutex persistentTypeIDMutex("Persistent Type ID Table Lock");  
#endif


//----------------------------------------------------------------------
PersistentTypeID::PersistentTypeID(const string& typeName, 
				   const string& parentName,
				   Persistent* (*maker)(),
				   Persistent* (*bc1)(),
				   Persistent* (*bc2)()) :  
  type(typeName),
  parent(parentName), 
  maker(maker),
  bc_maker1(bc1),
  bc_maker2(bc2)
{
#if DEBUG
  // Using printf as cerr causes a core dump (probably cerr has not
  // been initialized as this runs before main...)  (This may be Mac specific...)
  printf("PersistentTypeID constructor:\n");
  printf("   typename:   %s\n", typeName.c_str() );
  printf("   parentname: %s\n", parentName.c_str() );
  printf("   maker:      %p\n\n", maker );
#endif

  persistentTypeIDMutex.lock();
  if (!table)
  {
    table = scinew Piostream::MapStringPersistentTypeID;

#if DEBUG
    printf( "created table:  %p\n", table);
#endif
  }
#if DEBUG
  else
    printf( "table is:  %p\n", table);
#endif
  
  Piostream::MapStringPersistentTypeID::iterator dummy;
 
  dummy = table->find(type);

  if (dummy != table->end())
  {
    if ((*dummy).second->maker != maker 
	|| ((*dummy).second->parent != parentName))
    {
      printf( "WARNING: duplicate type in Persistent Object Type Database: %s\n",
              type.c_str() );
      persistentTypeIDMutex.unlock();
      return;
    }
  }
  
#if DEBUG
  printf("putting in table: PersistentTypeID: %s %s\n", typeName.c_str(), parentName.c_str() );
#endif

  (*table)[type] = this;
  persistentTypeIDMutex.unlock();
}


PersistentTypeID::~PersistentTypeID()
{
  Piostream::MapStringPersistentTypeID::iterator iter;

  if (table == NULL)
  {
    printf( "WARNING: Persistent.cc: ~PersistentTypeID(): table is NULL\n" );
    printf( "         For: %s, %s\n", type.c_str(), parent.c_str() );
    return;
  }

  iter = table->find(type);
  if (iter == table->end())
  {
    printf( "WARNING: Could not remove type from Object type database: %s\n",
            type.c_str() );
  }
  else
  {
    table->erase(iter);
  }

  if (table->size() == 0)
  {
    delete table;
    table = 0;
  }
}


//----------------------------------------------------------------------
Persistent::~Persistent()
{
}


// GROUP: Piostream class implementation
//////////
//

//----------------------------------------------------------------------
Piostream::Piostream(Direction dir, int version, const string &name,
                     ProgressReporter *pr)
  : dir(dir),
    version_(version),
    err(false),
    outpointers(0),
    inpointers(0),
    current_pointer_id(1),
    have_peekname_(false),
    reporter_(pr),
    own_reporter_(false),
    backwards_compat_id_(false),
    file_name(name)
{
  if (reporter_ == NULL)
  {
    reporter_ = scinew ProgressReporter();
    own_reporter_ = true;
  }
}

//----------------------------------------------------------------------
Piostream::~Piostream()
{
  if (own_reporter_) { delete reporter_; }
}

//----------------------------------------------------------------------
void
Piostream::emit_pointer(int& have_data, int& pointer_id)
{
  io(have_data);
  io(pointer_id);
}

//----------------------------------------------------------------------
string
Piostream::peek_class()
{
  have_peekname_ = true;
  io(peekname_);
  return peekname_;
}

//----------------------------------------------------------------------
int
Piostream::begin_class(const string& classname, int current_version)
{
  if (err) return -1;
  int version = current_version;
  string gname;
  if (dir == Write)
  {
    gname = classname;
    io(gname);
  }
  else if (dir == Read && have_peekname_)
  {
    gname = peekname_;
  }
  else
  {
    io(gname);
  }
  have_peekname_ = false;

//   if (dir == Read)
//   {
//     if (classname != gname)
//     {
//       err = true;
//       reporter_->error(string("Expecting class: ") + classname +
//                        ", got class: " + gname + ".");
//       return 0;
//     }
//   }

  io(version);

  if (dir == Read && version > current_version)
  {
    err = true;
    reporter_->error("File too new.  " + classname + " has version " +
                     to_string(version) +
                     ", but this scirun build is at version " +
                     to_string(current_version) + ".");
  }

  return version;
}

//----------------------------------------------------------------------
void
Piostream::end_class()
{
}

//----------------------------------------------------------------------
void
Piostream::begin_cheap_delim()
{
}

//----------------------------------------------------------------------
void
Piostream::end_cheap_delim()
{
}

//----------------------------------------------------------------------
void
Piostream::io(bool& data)
{
  if (err) return;
  unsigned char tmp = data;
  io(tmp);
  if (dir == Read)
  {
    data = tmp;
  }
}

//----------------------------------------------------------------------
static
PersistentTypeID*
find_derived( const string& classname, const string& basename )
{
#if DEBUG
  printf("looking for %s, %s\n", classname.c_str(), basename.c_str());
#endif
  persistentTypeIDMutex.lock();
#if DEBUG
  printf("table is: %p\n", table);
#endif
  if (!table) return 0;
  PersistentTypeID* pid;
  
  Piostream::MapStringPersistentTypeID::iterator iter;
  
  iter = table->find(classname);
  if(iter == table->end()) {
#if DEBUG
    printf("not found in table %p\n",table );


    cerr << "The contents of the PID table keys: " << endl;
    iter = table->begin();
    while (iter != table->end()) {
      cerr << (*iter).first << endl;
      ++iter;
    }
#endif

    persistentTypeIDMutex.unlock();
    return 0;
  }
  persistentTypeIDMutex.unlock();
  
  pid = (*iter).second;
  if( pid->parent.size() == 0 ) {
#if DEBUG
    printf("size is 0\n");
#endif
    return 0;
  }
  
  if (basename == pid->parent) return pid;
  
  if (find_derived(pid->parent, basename)) return pid;
  
  return 0;
}

//----------------------------------------------------------------------

void
Piostream::io(Persistent*& data, const PersistentTypeID& pid)
{
#if DEBUG
  printf("looking for pid: %s, %s\n", pid.type.c_str(), pid.parent.c_str() );
#endif
  if (err) return;
  if (dir == Read)
  {
    int have_data;
    int pointer_id;
    data = 0;
    emit_pointer(have_data, pointer_id);

#if DEBUG
    printf("after emit: %d, %d\n", have_data, pointer_id);
#endif

    if (have_data)
    {
      // See what type comes next in the stream.  If it is a type
      // derived from pid->type, then read it in, otherwise it is an
      // error.
      const string in_name(peek_class());
      const string want_name(pid.type);
      
#if DERIVED
      printf("in here: %s, %s\n", in_name.c_str(), want_name.c_str());
#endif

      Persistent* (*maker)() = 0;
      Persistent* (*bc_maker1)() = 0;
      Persistent* (*bc_maker2)() = 0;
      if (in_name == want_name || backwards_compat_id_)
      {
	maker = pid.maker;
      }
      else
      {
	PersistentTypeID* found_pid = find_derived(in_name, want_name);
	
	if (found_pid)
        {
	  maker = found_pid->maker;
	  bc_maker1 = found_pid->bc_maker1;
	  bc_maker2 = found_pid->bc_maker2;
	  if (bc_maker1) set_backwards_compat_id(true);
	}
        else
        {
#if DEBUG
	  reporter_->error("Did not find a pt_id.");
#endif
	}
      }
      if (!maker)
      {
	reporter_->error("Maker not found? (class=" + in_name + ").");
	reporter_->error("want_name: " + want_name + ".");
	err = true;
	return;
      }
      
      // Make it.
      data = (*maker)();
      // Read it in.
      data->io(*this);
      if (err && backwards_compat_id_) {
	err = 0;
	reset_post_header();
	// replicate the io that has gone before this point.
	begin_cheap_delim();
	int hd;
	int p_id;
	delete data;
	data = 0;
	emit_pointer(hd, p_id);
	if (hd) peek_class();
	data = (*bc_maker1)();
	// Read it in.
	data->io(*this);
	if (err && bc_maker2) {
	  err = 0;
	  reset_post_header();
	  // replicate the io that has gone before this point.
	  begin_cheap_delim();
	  int hd;
	  int p_id;
	  delete data;
	  data = 0;
	  emit_pointer(hd, p_id);
	  if (hd) peek_class();
	  data = (*bc_maker2)();
	  // Read it in.
	  data->io(*this);
	}
      }
    

      // Insert this pointer in the database.
      if (!inpointers)
      {
	inpointers = scinew MapIntPersistent;
      }
      (*inpointers)[pointer_id] = data;
    }
    else
    {
      // Look it up.
      if (pointer_id == 0)
      {
	data = 0;
      }
      else {
	MapIntPersistent::iterator initer;
	if (inpointers) initer = inpointers->find(pointer_id);
	if (!inpointers || initer == inpointers->end())
        {
	  reporter_->error("Pointer not in file, but should be!.");
	  err = true;
	  return;
	}
	data = (*initer).second;
      }
    }
  }
  else // dir == Write
  {		
    int have_data;
    int pointer_id;
    
    MapPersistentInt::iterator outiter;
    if (outpointers)
    {
      outiter = outpointers->find(data);
      pointer_id = (*outiter).second;
    }
    
    if (data == 0)
    {
      have_data = 0;
      pointer_id = 0;
    }
    else if (outpointers && outiter != outpointers->end())
    {
      // Already emitted, pointer id fetched from hashtable.
      have_data = 0;
    }
    else
    {
      // Emit it.
      have_data = 1;
      pointer_id = current_pointer_id++;
      if (!outpointers)
      {
	outpointers = scinew MapPersistentInt;
      }
      (*outpointers)[data] = pointer_id;
    }
    
    emit_pointer(have_data, pointer_id);
    
    if (have_data)
    {
      data->io(*this);
    }
  }
}


//----------------------------------------------------------------------
Piostream*
auto_istream(const string& filename, ProgressReporter *pr)
{
  std::ifstream in(filename.c_str());
  if (!in)
  {
    if (pr) pr->error("File not found: " + filename);
    else cerr << "ERROR - File not found: " << filename << endl;
    return 0;
  }

  // Create a header of size 16 to account for new endianness
  // flag in binary headers when the version > 1.
  char hdr[16]; 
  in.read(hdr, 16);

  if (!in)
  {
    if (pr) pr->error("Unable to open file: " + filename);
    else cerr << "ERROR - Unable to open file: " << filename << endl;
    return 0;
  }

  // Close the file.
  in.close();

  // Determine endianness of file.
  int file_endian, version;

  if (!Piostream::readHeader(pr, filename, hdr, 0, version, file_endian))
  {
    if (pr) pr->error("Cannot parse header of file: " + filename);
    else cerr << "ERROR - Cannot parse header of file: " << filename << endl;
    return 0;
  }
  if (version > Piostream::PERSISTENT_VERSION)
  {
    const string errmsg = "File '" + filename + "' has version " +
      to_string(version) + ", this build only supports up to version " +
      to_string(Piostream::PERSISTENT_VERSION) + ".";
    if (pr) pr->error(errmsg);
    else cerr << "ERROR - " + errmsg;
    return 0;
  }

  const char m1 = hdr[4];
  const char m2 = hdr[5];
  const char m3 = hdr[6];
  if (m1 == 'B' && m2 == 'I' && m3 == 'N')
  {
    // Old versions of Pio used XDR which always wrote big endian so if
    // the version = 1, readHeader would return BIG, otherwise it will
    // read it from the header.
    int machine_endian = Piostream::Big;
#ifdef HAVE_TEEM
    if (airMyEndian == airEndianLittle) 
#else
    if ( isLittleEndian() )
#endif
      machine_endian = Piostream::Little;

    if (file_endian == machine_endian) 
      return scinew BinaryPiostream(filename, Piostream::Read, version, pr);
    else 
      return scinew BinarySwapPiostream(filename, Piostream::Read, version,pr);
  }
  else if (m1 == 'A' && m2 == 'S' && m3 == 'C')
  {
    return scinew TextPiostream(filename, Piostream::Read, pr);
  }

  if (pr) pr->error(filename + " is an unknown type!");
  else cerr << filename << " is an unknown type!" << endl;
  return 0;
}


//----------------------------------------------------------------------
Piostream*
auto_ostream(const string& filename, const string& type, ProgressReporter *pr)
{
  // Based on the type string do the following
  //     Binary:  Return a BinaryPiostream 
  //     Fast:    Return FastPiostream
  //     Text:    Return a TextPiostream
  //     Default: Return BinaryPiostream 
  // NOTE: Binary will never return BinarySwap so we always write
  //       out the endianness of the machine we are on
  Piostream* stream;
  if (type == "Binary")
  {
    stream = scinew BinaryPiostream(filename, Piostream::Write, -1, pr);
  }
  else if (type == "Text")
  {
    stream = scinew TextPiostream(filename, Piostream::Write, pr);
  }
  else if (type == "Fast")
  {
    stream = scinew FastPiostream(filename, Piostream::Write, pr);
  }
  else
  {
    stream = scinew BinaryPiostream(filename, Piostream::Write, -1, pr);
  }
  return stream;
}


//----------------------------------------------------------------------
bool
Piostream::readHeader( ProgressReporter *pr,
                       const string & filename, char * hdr,
		       const char   * filetype, int  & version,
		       int & endian)
{
  char m1=hdr[0];
  char m2=hdr[1];
  char m3=hdr[2];
  char m4=hdr[3];

  if (m1 != 'S' || m2 != 'C' || m3 != 'I' || m4 != '\n')
  {
    if (pr)
    {
      pr->error( filename + " is not a valid SCI file! (magic=" +
                 m1 + m2 + m3 + m4 + ").");
    }
    else
    {
      cerr << filename << " is not a valid SCI file! (magic=" <<
        m1 << m2 << m3 << m4 << ")." << endl;
    }
    return false;
  }
  char v[5];
  v[0]=hdr[8];
  v[1]=hdr[9];
  v[2]=hdr[10];
  v[3]=hdr[11];
  v[4]=0;
  istringstream in(v);
  in >> version;
  if (!in)
  {
    if (pr)
    {
      pr->error("Error reading file: " + filename +
                " (while reading version).");
    }
    else
    {
      cerr << "Error reading file: " << filename <<
        " (while reading version)." << endl;
    }
    return false;
  }
  if (filetype)
  {
    if (hdr[4] != filetype[0] || hdr[5] != filetype[1] ||
        hdr[6] != filetype[2])
    {
      if (pr) pr->error("Wrong filetype: " + filename);
      else cerr << "Wrong filetype: " << filename << endl;
      return false;
    }
  }
  
  bool is_binary = false;
  if (hdr[4] == 'B' && hdr[5] == 'I' && hdr[6] == 'N' && hdr[7] == '\n')
    is_binary = true;
  if(version > 1 && is_binary) {
    // can only be BIG or LIT
    if (hdr[12] == 'B' && hdr[13] == 'I' && 
	hdr[14] == 'G' && hdr[15] == '\n') {
      endian = Big;
    } else if (hdr[12] == 'L' && hdr[13] == 'I' && 
	       hdr[14] == 'T' && hdr[15] == '\n') {
      endian = Little;
    } else {
      if (pr)
      {
        pr->error(string("Unknown endianness: ") +
                  hdr[12] + hdr[13] + hdr[14]);
      }
      else
      {
        cerr << "Unknown endianness: " <<
          hdr[12] << hdr[13] << hdr[14] << endl;
      }
      return false;
    }
  } else {
    endian = Big; // old system using XDR always read/wrote big endian
  }
  return true;
}

} // End namespace SCIRun


