#ifndef DIRENT_INCLUDED
#define DIRENT_INCLUDED

/*

    Declaration of POSIX directory browsing functions and types for Win32.

    Author:  Kevlin Henney (kevlin@acm.org, kevlin@curbralan.com)
    History: Created March 1997. Updated June 2003.
    Rights:  See end of file.
    
*/

/* modified by Bryan Worthen, only to add __declspec directives and remove the extern "C"*/
#include <Core/OS/share.h>


typedef struct DIR DIR;

struct dirent
{
    char *d_name;
};

SCISHARE DIR           *opendir(const char *);
SCISHARE int           closedir(DIR *);
SCISHARE struct dirent *readdir(DIR *);
SCISHARE void          rewinddir(DIR *);

/*

    Copyright Kevlin Henney, 1997, 2003. All rights reserved.

    Permission to use, copy, modify, and distribute this software and its
    documentation for any purpose is hereby granted without fee, provided
    that this copyright and permissions notice appear in all copies and
    derivatives.
    
    This software is supplied "as is" without express or implied warranty.

    But that said, if there are any problems please get in touch.

*/

#endif
