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


#ifndef UUID_MASTER
#define UUID_MASTER 8

#include <sci_defs/config_defs.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#if HAVE_SYS_UUID_H
   extern "C" { // SGI uuid.h doesn't have this, so we need extern C here
#  include <sys/uuid.h>
   }
#  define UUID_CREATE
#else
#  if HAVE_UUID_UUID_H
      extern "C" { // Linux uuid.h doesn't have this, so we need extern C here
#     include <uuid/uuid.h>
      }
#     define UUID_GENERATE
#  else
#     define GENERATE_CUSTOM_UUID
#     include <Core/Util/GenerateUUID.h>
#  endif
#endif

/*
 * Retreives a UUID
 * */
std::string getUUID()
{
#ifdef UUID_CREATE
  char* uuid_str;
  uuid_t uuid;
  uint_t status;

  uuid_create(&uuid, &status);
  if(status != uuid_s_ok){
    std::cerr << "Error creating uuid!\n";
    exit(1);
  }

  uuid_to_string(&uuid, &uuid_str, &status);
  if(status != uuid_s_ok){
    std::cerr << "Error creating uuid string!\n";
    exit(1);
  }
  return std::string(uuid_str);
#endif

#ifdef UUID_GENERATE
  char* uuid_str;
  uuid_t uuid;

  uuid_str = (char*)malloc(64*sizeof(char));
  uuid_generate( uuid );
  uuid_unparse(uuid, uuid_str);
  return std::string(uuid_str);
#endif

#ifdef GENERATE_CUSTOM_UUID
  return genUUID();
#endif

}


#endif
