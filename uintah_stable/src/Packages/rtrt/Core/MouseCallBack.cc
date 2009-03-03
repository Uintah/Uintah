/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/MouseCallBack.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace rtrt;

static map<const Object*, cbFunc> down_callbacks;
static map<const Object*, cbFunc> up_callbacks;
static map<const Object*, cbFunc> motion_callbacks;



void
MouseCallBack::assignCB_MD( cbFunc fp, const Object* obj )
{
  down_callbacks.insert(make_pair(obj, fp));
}

void
MouseCallBack::assignCB_MU( cbFunc fp, const Object* obj )
{
  up_callbacks.insert(make_pair(obj, fp));
}

void
MouseCallBack::assignCB_MM( cbFunc fp, const Object* obj )
{
  motion_callbacks.insert(make_pair(obj, fp));
}

bool
MouseCallBack::hasCB_MD( const Object* obj )
{
  return down_callbacks.find(obj) != down_callbacks.end();
}

bool
MouseCallBack::hasCB_MU( const Object* obj )
{
  return up_callbacks.find(obj) != up_callbacks.end();
}

bool
MouseCallBack::hasCB_MM( const Object* obj )
{
  return motion_callbacks.find(obj) != motion_callbacks.end();
}

cbFunc
MouseCallBack::getCB_MD( const Object* obj )
{
  map<const Object*, cbFunc>::iterator iter = down_callbacks.find(obj);
  if(iter == down_callbacks.end())
    return 0;
  else
    return iter->second;
}

cbFunc
MouseCallBack::getCB_MU( const Object* obj )
{
  map<const Object*, cbFunc>::iterator iter = up_callbacks.find(obj);
  if(iter == up_callbacks.end())
    return 0;
  else
    return iter->second;
}

cbFunc
MouseCallBack::getCB_MM( const Object* obj )
{
  map<const Object*, cbFunc>::iterator iter = motion_callbacks.find(obj);
  if(iter == motion_callbacks.end())
    return 0;
  else
    return iter->second;
}
