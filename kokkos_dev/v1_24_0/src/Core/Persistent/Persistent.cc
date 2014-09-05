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

#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace std;

#define DEBUG 0

namespace SCIRun {

static Piostream::MapStringPersistentTypeID* table = 0;  

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
				   Persistent* (*maker)())
  :  type(typeName), parent(parentName), maker(maker)
{
#if DEBUG
  cerr << "PersistentTypeID constructor: " << typeName << " " << parentName << ", maker: " << maker << "\n";
#endif

  persistentTypeIDMutex.lock();
  if (!table)
  {
    table = scinew Piostream::MapStringPersistentTypeID;

#if DEBUG
    cerr << "table " << &table << ", pointer is " << table << "\n";
#endif
  }
  
  Piostream::MapStringPersistentTypeID::iterator dummy;
 
  dummy = table->find(type);

  if (dummy != table->end())
  {
    if ((*dummy).second->maker != maker 
	|| ((*dummy).second->parent != parentName))
    {
      cerr << "WARNING: duplicate type in Persistent "
	   << "Object Type Database: " << type << endl;
      persistentTypeIDMutex.unlock();
      return;
    }
  }
  
#if DEBUG
  cerr << "putting in table: PersistentTypeID: " << typeName << " " << parentName << endl;
#endif

  (*table)[type] = this;
  persistentTypeIDMutex.unlock();
}


PersistentTypeID::~PersistentTypeID()
{
  Piostream::MapStringPersistentTypeID::iterator iter;

  if (table == NULL)
  {
    cerr << "WARNING: Persistent.cc: ~PersistentTypeID(): table is NULL\n";
    cerr << "         For: " << type << ", " << parent << "\n";
    return;
  }

  iter = table->find(type);
  if (iter == table->end())
  {
    cerr << "WARNING: Could not remove type from Object type database: " <<
      type << endl;
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
Piostream::Piostream(Direction dir, int version, const string &name)
: dir(dir), version(version), err(0), outpointers(0), inpointers(0),
  current_pointer_id(1), file_name(name)
{
}

//----------------------------------------------------------------------
Piostream::~Piostream()
{
}

//----------------------------------------------------------------------
int
Piostream::reading()
{
    return dir==Read;
}

//----------------------------------------------------------------------
int
Piostream::writing()
{
    return dir==Write;
}

//----------------------------------------------------------------------
int
Piostream::error()
{
    return err;
}

//----------------------------------------------------------------------
static
PersistentTypeID*
find_derived( const string& classname, const string& basename )
{
  persistentTypeIDMutex.lock();
#if DEBUG
  cout << "table is: " << table << "\n";
#endif
  if (!table) return 0;
  PersistentTypeID* pid;
  
  Piostream::MapStringPersistentTypeID::iterator iter;
  
  iter = table->find(classname);
  if(iter == table->end()) {
#if DEBUG
    cerr << "not found in table " << &table << ", pointer is " << table << "\n";
#endif
    persistentTypeIDMutex.unlock();
    return 0;
  }
  persistentTypeIDMutex.unlock();
  
  pid = (*iter).second;
  if( pid->parent.size() == 0 ) {
#if DEBUG
    cout << "size is 0\n";
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
  if (err) {
    return;
  }
  if (dir == Read) {
    int have_data;
    int pointer_id;
    data=0;			// In case anything goes wrong...
    emit_pointer(have_data, pointer_id);
    if (have_data) {
      // See what type comes next in the stream.  If it is a type
      // derived from pid->type, then read it in, otherwise it is an
      // error...
      string in_name(peek_class());
      string want_name(pid.type);
      
      Persistent* (*maker)() = 0;
      if (in_name == want_name) {
	maker=pid.maker;
      }
      else {
	PersistentTypeID* found_pid = find_derived(in_name, want_name);
	
	if (found_pid) {
	  maker=found_pid->maker;
	} else {
#if DEBUG
	  cerr << "Did not find a pt_id.\n";
#endif
	}
      }
      if (!maker) {
	cerr << "Maker not found? (class=" << in_name << ")\n";
	cerr << "want_name: " << want_name << "\n";
	err=1;
	return;
      }
      
				// Make it..
      data=(*maker)();
				// Read it in...
      data->io(*this);
      
				// Insert this pointer in the database
      if (!inpointers) {
	inpointers = scinew MapIntPersistent;
      }
      (*inpointers)[pointer_id] = data;
    } else {
				// Look it up...
      if (pointer_id==0) {
	data=0;
      }
      else {
	MapIntPersistent::iterator initer;
	if (inpointers) initer = inpointers->find(pointer_id);
	if (!inpointers || initer == inpointers->end()) {
	  cerr << "Error - pointer not in file, but should be!\n";
	  err=1;
	  return;
	}
	data = (*initer).second;
      }
    }
  }
  else {			// dir == Write
    int have_data;
    int pointer_id;
    
    MapPersistentInt::iterator outiter;
    if (outpointers) {
      outiter = outpointers->find(data);
      pointer_id = (*outiter).second;
    }
    
    if (data==0) {
      have_data=0;
      pointer_id=0;
    }
    else if (outpointers && outiter != outpointers->end()){
				// Already emitted, pointer id fetched
				// from hashtable
      have_data=0;
    }
    else {
				// Emit it..
      have_data=1;
      pointer_id=current_pointer_id++;
      if (!outpointers) {
				// scinew?
	outpointers = new MapPersistentInt;
      }
      (*outpointers)[data] = pointer_id;
    }
    
    emit_pointer(have_data, pointer_id);
    
    if(have_data) {
      data->io(*this);
    }
    
  }
}

//----------------------------------------------------------------------

Piostream*
auto_istream(const string& filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    cerr << "file not found: " << filename << endl;
    return 0;
  }
  char hdr[12];
  in.read(hdr, 12);
  if (!in) {
    cerr << "Error reading header of file: " << filename << "\n";
    return 0;
  }
  int version;
  if (!Piostream::readHeader(filename, hdr, 0, version)) {
    cerr << "Error parsing header of file: " << filename << "\n";
    return 0;
  }
  if(version != 1){
    cerr << "Unknown PIO version: " << version << ", found in file: " << filename << '\n';
    return 0;
  }
  char m1=hdr[4];
  char m2=hdr[5];
  char m3=hdr[6];
  if(m1 == 'B' && m2 == 'I' && m3 == 'N'){
    return scinew BinaryPiostream(filename, Piostream::Read);
  } else if(m1 == 'A' && m2 == 'S' && m3 == 'C'){
    return scinew TextPiostream(filename, Piostream::Read);
  } else if(m1 == 'G' && m2 == 'Z' && m3 == 'P'){
    return scinew GunzipPiostream(filename, Piostream::Read);
  } else {
    cerr << filename << " is an unknown type!\n";
    return 0;
  }
}

//----------------------------------------------------------------------
bool
Piostream::readHeader( const string & filename, char * hdr,
		       const char   * filetype, int  & version )
{
  char m1=hdr[0];
  char m2=hdr[1];
  char m3=hdr[2];
  char m4=hdr[3];
  if(m1 != 'S' || m2 != 'C' || m3 != 'I' || m4 != '\n') {
    cerr << filename << " is not a valid SCI file! (magic="
	 << m1 << m2 << m3 << m4 << ")\n";
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
  if(!in){
    cerr << "Error reading file: " << filename << " (while reading version)" << endl;
    return false;
  }
  if(filetype){
    if(hdr[4] != filetype[0] || hdr[5] != filetype[1] || hdr[6] != filetype[2]){
      cerr << "Wrong filetype: " << filename << endl;
      return false;
    }
  }
  return true;
}

} // End namespace SCIRun


