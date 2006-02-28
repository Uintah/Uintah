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
 *  Light.cc: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geom/Light.h>
#include <Core/Thread/MutexPool.h>

namespace SCIRun {

PersistentTypeID Light::type_id("Light", "Persistent", 0);

#define LIGHT_LOCK_POOL_SIZE 50
MutexPool light_lock_pool("LightObj pool", LIGHT_LOCK_POOL_SIZE);

static int light_lock_pool_hash(Light *ptr)
{
  long k = ((long)ptr) >> 2; // Disgard unused bits, word aligned pointers.
  return (int)((k^(3*LIGHT_LOCK_POOL_SIZE+1))%LIGHT_LOCK_POOL_SIZE);
}   

Light::Light(const string& name, bool on, bool transformed)
  : ref_cnt(0),
    lock(*(light_lock_pool.getMutex(light_lock_pool_hash(this)))),
    name(name), on(on), transformed( transformed )
{
}

Light::~Light()
{
}

#define LIGHT_VERSION 1

void Light::io(Piostream& stream)
{
    stream.begin_class("Light", LIGHT_VERSION);
    Pio(stream, name);
    stream.end_class();
}

void Pio(Piostream& stream, Light*& light)
{
    Persistent* tlight=light;
    stream.io(tlight, Light::type_id);
    if(stream.reading())
	light=(Light*)tlight;
}

} // End namespace SCIRun


