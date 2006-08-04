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


#include "config.h"
#include <string.h>  /* memcpy */
#include "formatseed.h"


/* Location and value test for network-order integer type sign byte. */
#define SIGN_BYTE(item) (*(unsigned char *)(item))
#define NEGATIVE(item) (SIGN_BYTE(item) > 127)


/* Network and host data sizes. */
static const size_t DATA_SIZES[2][ASFMT_SIMPLE_TYPE_COUNT] =
  {{1, 8, 4, 4, 4, 2, 4, 4, 2},
   {sizeof(char), sizeof(double), sizeof(float), sizeof(int), sizeof(long),
    sizeof(short), sizeof(unsigned int), sizeof(unsigned long),
    sizeof(unsigned short)}};


#ifndef ASFMT_HOST_USES_IEEE754_FP


/*
 * Copies #source# to #destination#, converting between IEEE and host floating-
 * point format.  #whatType# must be ASFMT_DOUBLE_TYPE or ASFMT_FLOAT_TYPE.
 * #sourceIsHostFormat# indicates whether the source is host or IEEE format.
 * The IEEE version of the data will be in big-endian byte order even if the
 * host machine is little-endian.  For IEEE 754 details, look at
 * http://www.research.microsoft.com/~hollasch/cgindex/coding/ieeefloat.html
 */
static void
ConvertIEEE(const void *destination,
            const void *source,
            ASFMT_DataTypes whatType,
            int sourceIsHostFormat) {

  #define DOUBLE_BIAS 1023
  #define QUAD_BIAS 16383
  #define SINGLE_BIAS 127

  struct DoublePrecision {
    unsigned sign : 1;
    unsigned exponent : 11;
    unsigned leading : 4;
    unsigned char mantissa[6];
  } doublePrecision;

  struct Expanded {
    unsigned char sign;
    int exponent;
    unsigned char mantissa[16];
  } expanded;

  struct QuadPrecision {
    unsigned sign : 1;
    unsigned exponent : 15;
    unsigned char mantissa[14];
  }; /* For future reference. */

  struct SinglePrecision {
    unsigned sign : 1;
    unsigned exponent : 8;
    unsigned leading : 7;
    unsigned char mantissa[2];
  } singlePrecision;

  double doubleValue;
  unsigned exponentBias;
  double factor;
  int i;
  size_t mantissaLength;

  if(whatType == ASFMT_DOUBLE_TYPE) {
    exponentBias = DOUBLE_BIAS;
    mantissaLength = sizeof(doublePrecision.mantissa) + 1;
    factor = 16.0; /* 2.0 ^ bitsize(doublePrecision.leading) */
  }
  else {
    exponentBias = SINGLE_BIAS;
    mantissaLength = sizeof(singlePrecision.mantissa) + 1;
    factor = 128.0; /* 2.0 ^ bitsize(singlePrecision.leading) */
  }

  if(sourceIsHostFormat) {

    doubleValue = (whatType == ASFMT_DOUBLE_TYPE) ?
                  *(double *)source : *(float *)source;
    expanded.sign = doubleValue < 0.0;
    if(expanded.sign) {
      doubleValue = -doubleValue;
    }
    expanded.exponent = 0;

    if(doubleValue != 0.0) {
      /* Determine the exponent value by iterative shifts (mult/div by 2) */
      while(doubleValue >= 2.0) {
        expanded.exponent += 1;
        doubleValue /= 2.0;
      }
      while(doubleValue < 1.0) {
        expanded.exponent -= 1;
        doubleValue *= 2.0;
      }
      expanded.exponent += exponentBias;
      doubleValue -= 1.0;
    }

    /* Set the bytes of the mantissa by iterative shift and truncate. */
    for(i = 0; i < 16; i++) {
      doubleValue *= factor;
      expanded.mantissa[i] = (int)doubleValue;
      doubleValue -= expanded.mantissa[i];
      factor = 256.0;
    }

    /* Pack the expanded version into the destination. */
    if(whatType == ASFMT_DOUBLE_TYPE) {
      memcpy(doublePrecision.mantissa,
             &expanded.mantissa[1],
             sizeof(doublePrecision.mantissa));
      doublePrecision.leading = expanded.mantissa[0];
      doublePrecision.exponent = expanded.exponent;
      doublePrecision.sign = expanded.sign;
      *(struct DoublePrecision *)destination = doublePrecision;
    }
    else {
      memcpy(singlePrecision.mantissa,
             &expanded.mantissa[1],
             sizeof(singlePrecision.mantissa));
      singlePrecision.leading = expanded.mantissa[0];
      singlePrecision.exponent = expanded.exponent;
      singlePrecision.sign = expanded.sign;
      *(struct SinglePrecision *)destination = singlePrecision;
    }

  }
  else {

    /* Unpack the source into the expanded version. */
    if(whatType == ASFMT_DOUBLE_TYPE) {
      doublePrecision = *(struct DoublePrecision *)source;
      expanded.sign = doublePrecision.sign;
      expanded.exponent = doublePrecision.exponent;
      expanded.mantissa[0] = doublePrecision.leading;
      memcpy(&expanded.mantissa[1],
             doublePrecision.mantissa,
             sizeof(doublePrecision.mantissa));
    }
    else {
      singlePrecision = *(struct SinglePrecision *)source;
      expanded.sign = singlePrecision.sign;
      expanded.exponent = singlePrecision.exponent;
      expanded.mantissa[0] = singlePrecision.leading;
      memcpy(&expanded.mantissa[1],
             singlePrecision.mantissa,
             sizeof(singlePrecision.mantissa));
    }

    /* Set mantissa by via shifts and adds; allow for denormalized values. */
    doubleValue = (expanded.exponent == 0) ? 0.0 : 1.0;

    for(i = 0; i < mantissaLength; i++) {
      doubleValue += (double)expanded.mantissa[i] / factor;
      factor *= 256.0;
    }

    /* Set the exponent by iterative mults/divs by 2. */
    if(expanded.exponent == 0)
      ; /* Nothing to do. */
    else if(expanded.exponent == (exponentBias * 2 + 1))
      /*
       * An exponent of all ones represents one of three things:
       *   Infinity: mantissa of all zeros
       *   Indeterminate: sign of 1, mantissa leading one followed by all zeros
       *   NaN: all other values
       * None of these can be reliably produced by C operations.  We might be
       * able to get Infinity by dividing by zero, but, on a non-IEEE machine,
       * we're more likely to cause some sort of floating-point exception.
       */
      ;
    else
      expanded.exponent -= exponentBias;

    if(expanded.exponent < 0) {
      for(i = expanded.exponent; i < 0; i++)
        doubleValue /= 2.0;
    }
    else {
      for(i = 0; i < expanded.exponent; i++)
        doubleValue *= 2.0;
    }

    if(expanded.sign)
      doubleValue *= -1.0;

    if(whatType == ASFMT_DOUBLE_TYPE)
      *(double *)destination = doubleValue;
    else
      *(float *)destination = doubleValue;

  }

}


#endif


/*
 * Copies the network integer of size #sourceSize# stored in #source# to the
 * #destinationSize#-long area #destination#, padding or truncating as needed.
 * #signedType# indicates whether or not the source is signed.
 */
static void
ResizeInt(void *destination,
          size_t destinationSize,
          const void *source,
          size_t sourceSize,
          int signedType) {
  if(destinationSize == sourceSize) {
    memcpy(destination, source, sourceSize);
  }
  else if(sourceSize > destinationSize) {
    /* Truncate high-order bytes. */
    memcpy(destination,
           (char *)source + (sourceSize - destinationSize),
           destinationSize);
    /* If necessary, flip the destination sign to match that of the source. */
    if(signedType && (NEGATIVE(source) != NEGATIVE(destination))) {
      SIGN_BYTE(destination) ^= 128;
    }
  }
  else {
    /* Pad with zeros or extend sign, as appropriate. */
    memset(destination,
           (signedType && NEGATIVE(source)) ? 0xff : 0,
           destinationSize);
    memcpy((char *)destination + (destinationSize - sourceSize),
           source,
           sourceSize);
  }
}


/*
 * Copies #length# bytes from #from# to #to# in reverse order.  Will work
 * properly if #from# and #to# are the same address.
 */
static void
ReverseBytes(void *to,
             const void *from,
             size_t length) {

  char charBegin;
  const char *fromBegin;
  const char *fromEnd;
  char *toBegin;
  char *toEnd;

  for(fromBegin = (const char *)from, fromEnd = fromBegin + length - 1,
        toBegin = (char *)to, toEnd = toBegin + length - 1;
      fromBegin <= fromEnd;
      fromBegin++, fromEnd--, toBegin++, toEnd--) {
    charBegin = *fromBegin;
    *toBegin = *fromEnd;
    *toEnd = charBegin;
  }

}


void
ASFMT_ConvertData(void *destination,
                  const void *source,
                  const ASFMT_DataDescriptor *description,
                  size_t length,
                  int sourceIsHostFormat) {

  size_t destStructSize;
  size_t networkBytesConverted;
  char *nextDest;
  const char *nextSource;
  size_t sourceStructSize;
  int structRepetitions;

  networkBytesConverted = 0;

  for( ; length > 0; length--, description++) {
    if(sourceIsHostFormat) {
      nextDest = (char *)destination + networkBytesConverted;
      nextSource = (char *)source + description->offset;
    }
    else {
      nextDest = (char *)destination + description->offset;
      nextSource = (char *)source + networkBytesConverted;
    }
    if(description->type == ASFMT_STRUCT_TYPE) {
      destStructSize = ASFMT_DataSize(description->members,
                                      description->length,
                                      !sourceIsHostFormat);
      sourceStructSize = ASFMT_DataSize(description->members,
                                        description->length,
                                        sourceIsHostFormat);
      if(sourceIsHostFormat) {
        sourceStructSize += description->tailPadding;
      }
      else {
        destStructSize += description->tailPadding;
      }
      for(structRepetitions = description->repetitions;
          structRepetitions-- > 0;
          nextDest += destStructSize,
          nextSource += sourceStructSize) {
        ASFMT_ConvertData(nextDest,
                          nextSource,
                          description->members,
                          description->length,
                          sourceIsHostFormat);
      }
    }
    else {
      ASFMT_HomogenousConvertData(nextDest,
                                  nextSource,
                                  description->type,
                                  description->repetitions,
                                  sourceIsHostFormat);
    }
    networkBytesConverted += ASFMT_NetworkDataSize(description, 1);
  }

}


size_t
ASFMT_DataSize(const ASFMT_DataDescriptor *description,
               size_t length,
               int hostFormat) {

  const ASFMT_DataDescriptor *lastMember;
  size_t totalSize;

  if(hostFormat) {
    lastMember = &description[length - 1];
    return lastMember->offset +
           ( (lastMember->type == ASFMT_STRUCT_TYPE) ?
             ( (ASFMT_HostDataSize(lastMember->members, lastMember->length) +
                lastMember->tailPadding) * lastMember->repetitions) :
             ASFMT_HomogenousHostDataSize(lastMember->type,
                                          lastMember->repetitions) );
  }
  else {
    totalSize = 0;
    for( ; length > 0; length--, description++) {
      totalSize += (description->type == ASFMT_STRUCT_TYPE) ?
        (ASFMT_NetworkDataSize(description->members, description->length) *
         description->repetitions) :
        ASFMT_HomogenousNetworkDataSize(description->type,
                                        description->repetitions);
    }
    return totalSize;
  }


}


int
ASFMT_DifferentFormat(ASFMT_DataTypes whatType) {

  #ifndef ASFMT_HOST_USES_IEEE754_FP

  if(whatType == ASFMT_DOUBLE_TYPE || whatType == ASFMT_FLOAT_TYPE) {
    double fpTester = -4.0;
    int littleEndian = ASFMT_DifferentOrder();
    unsigned char *bytes = (unsigned char *)&fpTester;
    /* Test sign, low-order bit of exponent and high-order bit of mantissa. */
    return bytes[littleEndian ? sizeof(fpTester) - 1 : 0] != 192 ||
           bytes[littleEndian ? sizeof(fpTester) - 2 : 1] !=
           ((sizeof(fpTester) == 4)  ? 128 :
            (sizeof(fpTester) == 8)  ? 16 :
            (sizeof(fpTester) == 16) ? 1 : 0);
  }

  #endif

  return 0;

}


int
ASFMT_DifferentOrder(void) {
  /* We could do this once during configuration, but the overhead is tiny. */
  int orderTester = 1;
  return SIGN_BYTE(&orderTester) != 0;
}


int
ASFMT_DifferentSize(ASFMT_DataTypes whatType) {
  return DATA_SIZES[0][whatType] != DATA_SIZES[1][whatType];
}


void
ASFMT_HomogenousConvertData(void *destination,
                            const void *source,
                            ASFMT_DataTypes whatType,
                            size_t repetitions,
                            int sourceIsHostFormat) {

  size_t destSize;
  int resize;
  int reverse;
  size_t sourceSize;

  destSize = DATA_SIZES[!sourceIsHostFormat][whatType];
  sourceSize = DATA_SIZES[sourceIsHostFormat][whatType];
  resize = destSize != sourceSize;
  reverse = ASFMT_DifferentOrder() && (destSize > 1);

#ifndef ASFMT_HOST_USES_IEEE754_FP
    if((whatType == ASFMT_DOUBLE_TYPE || whatType == ASFMT_FLOAT_TYPE) &&
       (ASFMT_DifferentFormat(whatType) || resize)) {
      for(;
          repetitions-- > 0;
          source = (char *)source + sourceSize,
          destination = (char *)destination + destSize) {
        ConvertIEEE(destination, source, whatType, sourceIsHostFormat);
      }
      /* Note: ConvertIEEE also handles byte ordering. */
      return;
    }
#endif

  if(resize) {
    int signedResize = whatType < ASFMT_UNSIGNED_INT_TYPE;
    if(reverse && sourceIsHostFormat) {
      char swapped[16];
      for(;
          repetitions-- > 0;
          source = (char *)source + sourceSize,
          destination = (char *)destination + destSize) {
        ReverseBytes(swapped, source, sourceSize);
        ResizeInt(destination, destSize, swapped, sourceSize, signedResize);
      }
    }
    else {
      for(;
          repetitions-- > 0;
          source = (char *)source + sourceSize,
          destination = (char *)destination + destSize) {
        ResizeInt(destination, destSize, source, sourceSize, signedResize);
        if(reverse) {
          ReverseBytes(destination, destination, destSize);
        }
      }
    }
  }
  else if(reverse) {
    for(;
        repetitions-- > 0;
        source = (char *)source + sourceSize,
        destination = (char *)destination + destSize) {
      ReverseBytes(destination, source, sourceSize);
    }
  }
  else if(destination != source) {
    memcpy(destination, source, sourceSize * repetitions);
  }

}


size_t
ASFMT_HomogenousDataSize(ASFMT_DataTypes whatType,
                         size_t repetitions,
                         int hostFormat) {
  return DATA_SIZES[hostFormat][whatType] * repetitions;
}
