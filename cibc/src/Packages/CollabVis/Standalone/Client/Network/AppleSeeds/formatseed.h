/* $Id$ */


/*
 * Copyright © 2000 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice, this paragraph and the following three paragraphs
 * appear in all copies. 
 *
 * Permission to incorporate this software into commercial products may be
 * obtained by contacting
 * Eric Lund
 * Technology Transfer Office 
 * 9500 Gilman Drive 
 * 411 University Center 
 * University of California 
 * La Jolla, CA 92093-0093
 * (858) 534-0175
 * ericlund@ucsd.edu
 *
 * This software program and documentation are copyrighted by The Regents of
 * the University of California. The software program and documentation are
 * supplied "as is", without any accompanying services from The Regents. The
 * Regents does not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the program was
 * developed for research purposes and is advised not to rely exclusively on
 * the program for any reason. 
 *
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED
 * HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO
 * OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. 
 */


#ifndef FORMATSEED_H
#define FORMATSEED_H


/*
 * This package provides facilities for translating between host and network
 * data formats.  It may be considered an extention of the Unix {hn}to{nh}*
 * functions or as a lightweight version of XDR.  It handles big-to-little
 * endian translations, integer size discrepencies, and conversions to/from
 * IEEE floating point format.  It does *not* handle either non-two's
 * complement integer formats or mixed-endian (e.g., little endian bytes, big
 * endian words) ordering.
 */


#include <stddef.h>    /* offsetof */
#include <sys/types.h> /* size_t */


#ifndef NULL
#  define NULL 0
#endif


#ifdef __cplusplus
extern "C" {
#endif


/*
 * Supported data types.  Network formats for the integer types are all two's
 * complement, most significant byte first; the sizes are 1, 2, 4, and 4 bytes,
 * respectively, for characters, shorts, ints, and longs.  float and double use
 * the four-byte and eight-byte IEEE 754 formats, respectively.  These match
 * the XDR specificiation (no support is presently provided for eight byte
 * ints) and differ from Java only in the size of chars (1 vs. 2) and longs
 * (4 vs. 8).  The STRUCT_TYPE allows C structs to be translated between
 * network and host formats with a single call.
 */
typedef enum {
    ASFMT_CHAR_TYPE, ASFMT_DOUBLE_TYPE, ASFMT_FLOAT_TYPE, ASFMT_INT_TYPE,
    ASFMT_LONG_TYPE, ASFMT_SHORT_TYPE, ASFMT_UNSIGNED_INT_TYPE,
    ASFMT_UNSIGNED_LONG_TYPE, ASFMT_UNSIGNED_SHORT_TYPE, ASFMT_STRUCT_TYPE
  } ASFMT_DataTypes;
/* Count of the non-struct types. */
#define ASFMT_SIMPLE_TYPE_COUNT 9


/*
 * A description of a collection of data.  #type# indicates the data type.
 * #repetitions# is used only for arrays; it contains the number of elements.
 * #offset# is used only for struct members in host format; it contains the
 * offset of the member from the beginning of the struct, taking into account
 * internal padding added by the compiler for alignment purposes.  #members#,
 * #length#, and #tailPadding# are used only for ASFMT_STRUCT_TYPE data; the
 * #length#-long array #members# describes the members of the struct, and
 * #tailPadding# indicates how many padding bytes the compiler adds to the end
 * of the structure.
 */
typedef struct ASFMT_DataDescriptorStruct {
  ASFMT_DataTypes type;
  size_t repetitions;
  size_t offset;
  struct ASFMT_DataDescriptorStruct *members;
  size_t length;
  size_t tailPadding;
} ASFMT_DataDescriptor;

/*
 * A convenience macro for initializing an ASFMT_DataDescriptor for a data item
 * that is not a struct or struct member.
 */
#define ASFMT_SIMPLE_DATA(type,repetitions) {type, repetitions, 0, NULL, 0, 0}
/*
 * A convenience macro for initializing an ASFMT_DataDescriptor for a data item
 * that is a member of a struct.
 */
#define ASFMT_SIMPLE_MEMBER(type,repetitions,offset) \
  {type, repetitions, offset, NULL, 0, 0}
/*
 * A convenience macro for calculating the number of padding bytes added by the
 * compiler to the end of a struct.  Compilers add such padding in order to
 * properly align the elements in arrays of structs.  #structType# is the name
 * of the struct type, #lastMember# the name of its last member, #memberType#
 * its type, and #memberRepetitions# the number of elements in it if it is an
 * array. 
 */
#define ASFMT_PAD_BYTES(structType,lastMember,memberType,memberRepetitions) \
  sizeof(structType) - offsetof(structType, lastMember) - \
  sizeof(memberType) * memberRepetitions


/*
 * Translates data between host and network formats.  Translates the data
 * pointed to by #source# between host and network formats and stores the
 * result in #destination#.  The contents of #source# are described by the
 * #length#-long array #description#, and #sourceIsHostFormat# indicates
 * whether we're translating from host format to network format or vice versa.
 * The caller must insure that the memory pointed to by #destination# is of
 * sufficient size to contain the translated data.
 */
void
ASFMT_ConvertData(void *destination,
                  const void *source,
                  const ASFMT_DataDescriptor *description,
                  size_t length,
                  int sourceIsHostFormat);
/* A convenience macro for converting host data to network format. */
#define ASFMT_ConvertHostToNetwork(destination,source,description,length) \
  ASFMT_ConvertData(destination, source, description, length, 1)
/* A convenience macro for converting network data to host format. */
#define ASFMT_ConvertNetworkToHost(destination,source,description,length) \
  ASFMT_ConvertData(destination, source, description, length, 0)


/*
 * Returns the number of bytes required for a collection of data.  Returns the
 * number of bytes required to hold the objects indicated by the data described
 * by the #length#-long array #description#.  #hostFormat# indicates whether
 * the host or network format size is desired.
 */
size_t
ASFMT_DataSize(const ASFMT_DataDescriptor *description,
               size_t length,
               int hostFormat);
/* A convenience macro for determining the size of host data. */
#define ASFMT_HostDataSize(description,length) \
  ASFMT_DataSize(description, length, 1)
/* A convenience macro for determining the size of network data. */
#define ASFMT_NetworkDataSize(description,length) \
  ASFMT_DataSize(description, length, 0)


/*
 * Indicates whether host data format differs from network format.  Returns 1
 * or 0 depending on whether or not the host format for #whatType# differs from
 * the network format.
 */
int
ASFMT_DifferentFormat(ASFMT_DataTypes whatType);


/*
 * Indicates whether host byte order differs from network order.  Returns 1 or
 * 0 depending on whether or not the host architecture stores data in
 * little-endian (least significant byte first) order.
 */
int
ASFMT_DifferentOrder(void);


/*
 * Indicates whether host data size differs from network size.  Returns 1 or 0
 * depending on whether or not the host data size for #whatType# differs from
 * the network data size.
 */
int
ASFMT_DifferentSize(ASFMT_DataTypes whatType);


/*
 * A convenience function for converting homogenous data.  Performs the same
 * function as ASFMT_ConvertData on a block of #repetitions# occurrences of
 * #whatType# data.
 */
void
ASFMT_HomogenousConvertData(void *destination,
                            const void *source,
                            ASFMT_DataTypes whatType,
                            size_t repetitions,
                            int sourceIsHostFormat);
/*
 * A convenience macro for converting homogenous host data to network format.
 */
#define ASFMT_HomogenousConvertHostToNetwork(dest,source,whatType,repetitions) \
  ASFMT_HomogenousConvertData(dest, source, whatType, repetitions, 1)
/*
 * A convenience macro for converting homogenous network data to host format.
 */
#define ASFMT_HomogenousConvertNetworkToHost(dest,source,whatType,repetitions) \
  ASFMT_HomogenousConvertData(dest, source, whatType, repetitions, 0)

/*
 * A convenience function for sizing homogenous data.  Performs the same
 * function as ASFMT_DataSize on a block of #repetitions# occurrences of
 * #whatType# data.
 */
size_t
ASFMT_HomogenousDataSize(ASFMT_DataTypes whatType,
                         size_t repetitions,
                         int hostFormat);
/*
 * A convenience macro for determining the size of homogeneous host data.
 */
#define ASFMT_HomogenousHostDataSize(whatType,repetitions) \
  ASFMT_HomogenousDataSize(whatType, repetitions, 1)
/*
 * A convenience macro for determining the size of homogeneous network data.
 */
#define ASFMT_HomogenousNetworkDataSize(whatType,repetitions) \
  ASFMT_HomogenousDataSize(whatType, repetitions, 0)


#ifdef ASFMT_SHORT_NAMES

#define CHAR_TYPE ASFMT_CHAR_TYPE
#define DOUBLE_TYPE ASFMT_DOUBLE_TYPE
#define FLOAT_TYPE ASFMT_FLOAT_TYPE
#define INT_TYPE ASFMT_INT_TYPE
#define LONG_TYPE ASFMT_LONG_TYPE
#define SHORT_TYPE ASFMT_SHORT_TYPE
#define UNSIGNED_INT_TYPE ASFMT_UNSIGNED_INT_TYPE
#define UNSIGNED_LONG_TYPE ASFMT_UNSIGNED_LONG_TYPE
#define UNSIGNED_SHORT_TYPE ASFMT_UNSIGNED_SHORT_TYPE
#define STRUCT_TYPE ASFMT_STRUCT_TYPE
#define DataTypes ASFMT_DataTypes
#define SIMPLE_TYPE_COUNT ASFMT_SIMPLE_TYPE_COUNT
#define DataDescriptor ASFMT_DataDescriptor
#define SIMPLE_DATA ASFMT_SIMPLE_DATA
#define SIMPLE_MEMBER ASFMT_SIMPLE_MEMBER
#define PAD_BYTES ASFMT_PAD_BYTES

#define ConvertData ASFMT_ConvertData
#define ConvertHostToNetwork ASFMT_ConvertHostToNetwork
#define ConvertNetworkToHost ASFMT_ConvertNetworkToHost
#define DataSize ASFMT_DataSize
#define HostDataSize ASFMT_HostDataSize
#define NetworkDataSize ASFMT_NetworkDataSize
#define DifferentFormat ASFMT_DifferentFormat
#define DifferentOrder ASFMT_DifferentOrder
#define DifferentSize ASFMT_DifferentSize
#define HomogenousConvertHostToNetwork ASFMT_HomogenousConvertHostToNetwork
#define HomogenousConvertNetworkToHost ASFMT_HomogenousConvertNetworkToHost
#define HomogenousDataSize ASFMT_HomogenousDataSize
#define HomogenousHostDataSize ASFMT_HomogenousHostDataSize
#define HomogenousNetworkDataSize ASFMT_HomogenousNetworkDataSize

#endif


#ifdef __cplusplus
}
#endif


#endif
