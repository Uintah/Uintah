
#include <iostream>
#include <map>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/MouseCallBack.h>
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
