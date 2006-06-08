/* config.h.  Generated automatically by configure.  */
/* config.hin.  Generated automatically from configure.in by autoheader.  */
/* acconfig.h -- autoheader configuration file

   This file is part of the LZO real-time data compression library.

   Copyright (C) 1996-1999 Markus Franz Xaver Johannes Oberhumer

   The LZO library is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of
   the License, or (at your option) any later version.

   The LZO library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the LZO library; see the file COPYING.
   If not, write to the Free Software Foundation, Inc.,
   59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

   Markus F.X.J. Oberhumer
   markus.oberhumer@jk.uni-linz.ac.at
 */


#ifndef __LZO_CONFIG_H
#define __LZO_CONFIG_H

/* $TOP$ */

/* Define to empty if the keyword does not work.  */
/* #undef const */

/* Define if your C compiler doesn't accept -c and -o together.  */
/* #undef NO_MINUS_C_MINUS_O */

/* Define to `unsigned' if <sys/types.h> doesn't define.  */
/* #undef size_t */

/* Define if you have the ANSI C header files.  */
#define STDC_HEADERS 1

/* Define if you can safely include both <sys/time.h> and <time.h>.  */
#define TIME_WITH_SYS_TIME 1

/* Define if your machine can copy aligned words much faster than bytes.  */
/* #undef LZO_ALIGNED_OK_4 */

/* Define for machines where assembler versions are not available.  */
/* #undef LZO_NO_ASM */

/* Define for machines that can access unaligned short words.  */
#define LZO_UNALIGNED_OK_2 1

/* Define for machines that can access unaligned words.  */
#define LZO_UNALIGNED_OK_4 1

/* Define to your architecture name.  */
#define MFX_ARCH "i386"

/* Define for machines where ".align 4" means align to a 4 byte boundary.  */
/* #undef MFX_ASM_ALIGN_BYTES */

/* Define for machines where ".align 4" means align to a 2**4 boundary.  */
/* #undef MFX_ASM_ALIGN_PTWO */

/* Define for machines where the assmbler understands ".type".  */
/* #undef MFX_ASM_HAVE_TYPE */

/* Define for machines where global symbols don't have leading underscores.  */
/* #undef MFX_ASM_NAME_NO_UNDERSCORES */

/* Define to your byte order.  */
#define MFX_BYTE_ORDER 1234

/* Define to your CPU name.  */
#define MFX_CPU "i686"

/* Define if your memcmp is broken.  */
/* #undef NO_MEMCMP */

/* Define to `long' if <stddef.h> doesn't define.  */
/* #undef ptrdiff_t */

/* The number of bytes in a ptrdiff_t.  */
#define SIZEOF_PTRDIFF_T 4

/* The number of bytes in a size_t.  */
#define SIZEOF_SIZE_T 4

/* The number of bytes in a char *.  */
#define SIZEOF_CHAR_P 4

/* The number of bytes in a unsigned.  */
#define SIZEOF_UNSIGNED 4

/* The number of bytes in a unsigned long.  */
#define SIZEOF_UNSIGNED_LONG 4

/* The number of bytes in a unsigned short.  */
#define SIZEOF_UNSIGNED_SHORT 2

/* Define if you have the atol function.  */
#define HAVE_ATOL 1

/* Define if you have the fstat function.  */
#define HAVE_FSTAT 1

/* Define if you have the getrusage function.  */
#define HAVE_GETRUSAGE 1

/* Define if you have the gettimeofday function.  */
#define HAVE_GETTIMEOFDAY 1

/* Define if you have the lstat function.  */
#define HAVE_LSTAT 1

/* Define if you have the memcmp function.  */
#define HAVE_MEMCMP 1

/* Define if you have the memcpy function.  */
#define HAVE_MEMCPY 1

/* Define if you have the memicmp function.  */
/* #undef HAVE_MEMICMP */

/* Define if you have the memmove function.  */
#define HAVE_MEMMOVE 1

/* Define if you have the memset function.  */
#define HAVE_MEMSET 1

/* Define if you have the stat function.  */
#define HAVE_STAT 1

/* Define if you have the strcasecmp function.  */
#define HAVE_STRCASECMP 1

/* Define if you have the stricmp function.  */
/* #undef HAVE_STRICMP */

/* Define if you have the strncasecmp function.  */
#define HAVE_STRNCASECMP 1

/* Define if you have the strnicmp function.  */
/* #undef HAVE_STRNICMP */

/* Define if you have the times function.  */
#define HAVE_TIMES 1

/* Define if you have the <assert.h> header file.  */
#define HAVE_ASSERT_H 1

/* Define if you have the <ctype.h> header file.  */
#define HAVE_CTYPE_H 1

/* Define if you have the <endian.h> header file.  */
#define HAVE_ENDIAN_H 1

/* Define if you have the <limits.h> header file.  */
#define HAVE_LIMITS_H 1

/* Define if you have the <malloc.h> header file.  */
#define HAVE_MALLOC_H 1

/* Define if you have the <memory.h> header file.  */
#define HAVE_MEMORY_H 1

/* Define if you have the <signal.h> header file.  */
#define HAVE_SIGNAL_H 1

/* Define if you have the <stddef.h> header file.  */
#define HAVE_STDDEF_H 1

/* Define if you have the <sys/endian.h> header file.  */
/* #undef HAVE_SYS_ENDIAN_H */

/* Define if you have the <sys/param.h> header file.  */
#define HAVE_SYS_PARAM_H 1

/* Define if you have the <sys/resource.h> header file.  */
#define HAVE_SYS_RESOURCE_H 1

/* Define if you have the <sys/time.h> header file.  */
#define HAVE_SYS_TIME_H 1

/* Define if you have the <sys/times.h> header file.  */
#define HAVE_SYS_TIMES_H 1

/* Define if you have the <time.h> header file.  */
#define HAVE_TIME_H 1

/* Define if you have the <unchecked.h> header file.  */
/* #undef HAVE_UNCHECKED_H */

/* Define if you have the <unistd.h> header file.  */
#define HAVE_UNISTD_H 1

/* Define if you have the <zlib.h> header file.  */
#define HAVE_ZLIB_H 1

/* Define if you have the z library (-lz).  */
/* #undef HAVE_LIBZ */

/* Name of package */
#define PACKAGE "lzo"

/* Version number of package */
#define VERSION "1.06"

/* Define if using the dmalloc debugging malloc package */
/* #undef WITH_DMALLOC */


/* $BOTTOM$ */

#if defined(HAVE_SYS_RESOURCE_H) && !defined(TIME_WITH_SYS_TIME)
#  undef /**/ HAVE_SYS_RESOURCE_H
#endif

#if defined(HAVE_SYS_TIMES_H) && !defined(TIME_WITH_SYS_TIME)
#  undef /**/ HAVE_SYS_TIMES_H
#endif

#if !defined(HAVE_LIBZ)
#  undef /**/ HAVE_ZLIB_H
#endif

#if defined(NO_MEMCMP)
#  undef /**/ HAVE_MEMCMP
#endif

#if (SIZEOF_CHAR_P <= 0)
#  undef /**/ SIZEOF_CHAR_P
#endif

#if (SIZEOF_PTRDIFF_T <= 0)
#  undef /**/ SIZEOF_PTRDIFF_T
#endif

#if (SIZEOF_UNSIGNED <= 0)
#  undef /**/ SIZEOF_UNSIGNED
#endif

#if (SIZEOF_UNSIGNED_LONG <= 0)
#  undef /**/ SIZEOF_UNSIGNED_LONG
#endif

#if (SIZEOF_UNSIGNED_SHORT <= 0)
#  undef /**/ SIZEOF_UNSIGNED_SHORT
#endif

#if (SIZEOF_SIZE_T <= 0)
#  undef /**/ SIZEOF_SIZE_T
#endif

#endif /* already included */

/*
vi:ts=4
*/
