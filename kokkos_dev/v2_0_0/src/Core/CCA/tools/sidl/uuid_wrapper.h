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

#ifndef UUID_MASTER
#define UUID_MASTER 8

#include <sci_config.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#if HAVE_SYS_UUID_H
extern "C" { // SGI uuid.h doesn't have this, so we need extern C here
#include <sys/uuid.h>
}
#define UUID_CREATE
#else
#if HAVE_UUID_UUID_H
extern "C" { // Linux uuid.h doesn't have this, so we need extern C here
#include <uuid/uuid.h>
}
#define UUID_GENERATE
#else
#error We need either sys/uuid.h or uuid/uuid.h
#endif
#endif

/*
 * Retreives a UUID
 * */
std::string getUUID()
{
  char* uuid_str;
  uuid_t uuid;

#ifdef UUID_CREATE
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
#endif
#ifdef UUID_GENERATE
  uuid_str = (char*)malloc(64*sizeof(char));
  uuid_generate( uuid );
  uuid_unparse(uuid, uuid_str);
#endif

  return std::string(uuid_str);
}


#endif
