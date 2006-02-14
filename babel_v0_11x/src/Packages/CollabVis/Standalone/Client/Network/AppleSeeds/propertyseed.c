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
#include <ctype.h>     /* isspace */
#include <stdio.h>     /* file functions */
#include <stdlib.h>    /* free malloc realloc {un}setenv */
#include <string.h>    /* string manipulation functions */
#define ASPROP_SHORT_NAMES
#include "propertyseed.h"


/*
 * Basic PropertyList manipulation.  Note: Where available, we use {un}setenv
 * on the environ variable just in case some system implements environ weirdly.
 */
extern char **environ;


#ifndef NULL
#  define NULL 0
#endif


/*
 * Searches #list# for a property with a name matching the first #nameLen#
 * characters of #name#.  Returns a pointer to the property if found, else NULL.
 */
static Property *
PropertySearch(const PropertyList list,
               const char *name,
               size_t nameLen) {
  Property *property;
  ForEachProperty(list, property) {
    if(strncmp(*property, name, nameLen) == 0 && (*property)[nameLen] == '=')
      return property;
  }
  return NULL;
}

            
const Property
FindPropertyByName(const PropertyList list,
                   const char *name) {
  return FindPropertyByNameSized(list, name, strlen(name));
}


const Property
FindPropertyByNameSized(const PropertyList list,
                        const char *name,
                        size_t size) {
  Property *property = PropertySearch(list, name, size);
  return (property == NULL) ? NO_PROPERTY : *property;
}


void
FreePropertyList(PropertyList *list) {
  Property *property;
  ForEachProperty(*list, property)
    free(*property);
  free(*list);
  *list = NO_PROPERTY_LIST;
}


PropertyList
NewPropertyList(void) {
  PropertyList returnValue;
  returnValue = malloc(sizeof(Property));
  *returnValue = NO_PROPERTY;
  return returnValue;
}


size_t
PropertyCount(const PropertyList list) {
  Property *property;
  ForEachProperty(list, property)
    ; /* Nothing more to do. */
  return property - list;
}


const char *
PropertyName(const Property property) {
  size_t nameLen;
  static char *returnValue = NULL;
  if(property == NO_PROPERTY)
    return NULL;
  nameLen = strchr(property, '=') - property;
  returnValue = realloc(returnValue, nameLen + 1);
  strncpy(returnValue, property, nameLen);
  returnValue[nameLen] = '\0';
  return returnValue;
}


const char *
PropertyValue(const Property property) {
  return (property == NO_PROPERTY) ? NULL : (strchr(property, '=') + 1);
}


void
RemoveProperty(PropertyList *list,
               const char *name) {
  Property *property;
#ifdef ASPROP_HAVE_UNSETENV
  if(*list == environ) {
    unsetenv((char *)name);
    return;
  }
#endif
  if((property = PropertySearch(*list, name, strlen(name))) != NULL) {
    free(*property);
    for( ; *property != NO_PROPERTY; property++)
      *property = *(property + 1);
  }
}


void
SetProperty(PropertyList *list,
            const char *name,
            const char *value) {
  size_t listLen;
  Property newProperty;
  Property *oldProperty;
#ifdef ASPROP_HAVE_UNSETENV
  if(*list == environ) {
    setenv((char *)name, (char *)value, 1);
    return;
  }
#endif
  newProperty = (Property)malloc(strlen(name) + 1 + strlen(value) + 1);
  sprintf(newProperty, "%s=%s", name, value);
  if((oldProperty = PropertySearch(*list, name, strlen(name))) == NULL) {
    listLen = PropertyCount(*list);
    *list = realloc(*list, (listLen + 2) * sizeof(Property));
    (*list)[listLen] = newProperty;
    (*list)[listLen + 1] = NO_PROPERTY;
  }
  else {
    free(*oldProperty);
    *oldProperty = newProperty;
  }
}


/* PropertyList manipulation. */


PropertyList
PropertySublistByPrefix(const PropertyList list,
                        const char *prefix) {

  char *equals;
  size_t prefixLen = strlen(prefix);
  const Property *property;
  char *toCopy;
  PropertyList returnValue = NewPropertyList();

  ForEachProperty(list, property) {
    if(strncmp(*property, prefix, prefixLen) == 0) {
      toCopy = strdup(*property);
      equals = strchr(toCopy, '=');
      *equals = '\0';
      SetProperty(&returnValue, toCopy, equals + 1);
      free(toCopy);
    }
  }

  return returnValue;

}


char *
StringFromPropertyList(const PropertyList list) {

  const char *c;
  char *p;
  const Property *property;
  char *returnValue;
  size_t totalSize = 0;

  ForEachProperty(list, property) {
    for(c = *property; *c != '\0'; c++) {
      totalSize++;
      if(*c == '\\' || *c == '\n')
        totalSize++;
    }
  }

  returnValue = (char *)malloc(totalSize + 1);
  p = returnValue;

  ForEachProperty(list, property) {
    for(c = *property; *c != '\0'; c++) {
      if(*c == '\\' || *c == '\n')
        *p++ = '\\';
      *p++ = *c;
    }
    *p++ = '\n';
  }

  *p = '\0';
  return returnValue;

}


PropertyList
StringToPropertyList(const char *s) {

  const char *endProperty;
  char *equals;
  char *p;
  char *property = (char *)malloc(1);
  PropertyList returnValue = NewPropertyList();

  while(*s != '\0') {
    for(endProperty = strchr(s, '\n');
        *(endProperty - 1) == '\\';
        endProperty = strchr(endProperty + 1, '\n'))
      ; /* Nothing more to do. */
    property = realloc(property, endProperty - s + 1);
    for(p = property; s < endProperty; p++, s++) {
      if(*s == '\\' && (*(s + 1) == '\\' || *(s + 1) == '\n'))
        s++;
      *p = *s;
    }
    *p = '\0';
    equals = strchr(property, '=');
    *equals = '\0';
    SetProperty(&returnValue, property, equals + 1);
    s++;
  }

  free(property);
  return returnValue;

}


/* Property reference resolution. */


/*
 * Reallocates #s# in order to append the #len#-long text #toAppend# to it and
 * returns the result.
 */
static char *
ExtendString(char *s,
             const char *toAppend,
             size_t len) {
  int oldLen = strlen(s);
  char *returnValue = realloc(s, oldLen + len + 1);
  strncpy(returnValue + oldLen, toAppend, len);
  returnValue[oldLen + len] = '\0';
  return returnValue;
}


const char *
ResolveReferences(PropertyList properties,
                  const char *text,
                  const char *defaultValue) {

  size_t nameLen;
  const char *nameStart;
  const char *refEnd;
  const char *refStart;
  static char *result = NULL;
  const char *value;

  if(result == NULL)
    result = (char *)malloc(1);
  *result = '\0';

  for(refStart = strchr(text, '$'), refEnd = text;
      refStart != NULL;
      refStart = strchr(refEnd, '$')) {

    /* Copy any interspersed non-reference text. */
    if(refStart > refEnd)
      result = ExtendString(result, refEnd, refStart - refEnd);

    /* Find the end of the reference and the start and length of the name. */
    if(*(refStart + 1) == '{' || *(refStart + 1) == '(') {
      nameStart = refStart + 2;
      refEnd = strchr(nameStart, (*(refStart + 1) == '{') ? '}' : ')');
      if(refEnd == NULL)
        break; /* Bad reference. */
      refEnd++;
      nameLen = refEnd - nameStart - 1;
    }
    else {
      nameStart = refStart + 1;
      for(refEnd = nameStart;
          isalnum((int)*refEnd) || *refEnd == '_' || *refEnd == '.';
          refEnd++)
        ; /* Nothing more to do. */
      nameLen = refEnd - nameStart;
    }

    /* Copy the value, if there is one. */
    value = FindPropertyValueByNameSized(properties, nameStart, nameLen);
    result = (value != NULL) ?
             ExtendString(result, value, strlen(value)) :
             (defaultValue != NULL) ?
             ExtendString(result, defaultValue, strlen(defaultValue)) :
             ExtendString(result, refStart, refEnd - refStart);

  }

  /* Copy any trailing non-reference text. */
  if(*refEnd != '\0')
    result = ExtendString(result, refEnd, strlen(refEnd));

  return result;

}


void
ResolveReferencesInList(PropertyList *list,
                        const char *defaultValue) {
  Property *property;
  const char *resolved;
  const char *value;
  ForEachProperty(*list, property) {
    value = PropertyValue(*property);
    resolved = ResolveReferences(*list, value, defaultValue);
    if(strcmp(value, resolved) != 0)
      SetProperty(list, PropertyName(*property), resolved);
  }
}


/* Reading and writing property lists. */


static FILE **inputFiles = NULL;
static int inputFilesCount = 0;


static void
AddInputFile(FILE *file) {
  inputFiles = realloc(inputFiles, (inputFilesCount + 1) * sizeof(FILE *));
  inputFiles[inputFilesCount++] = file;
}


static int
GetCh(void) {
  int ch;
  while((ch = fgetc(inputFiles[inputFilesCount - 1])) == EOF &&
        inputFilesCount > 1)
    fclose(inputFiles[--inputFilesCount]);
  return ch;
}


static void
UngetCh(int ch) {
  if(ch != EOF)
    ungetc(ch, inputFiles[inputFilesCount - 1]);
}


/*
 * Places in #toWhere# a word that begins with #ch# and continues with
 * read characters up to the first occurrence of any character in
 * #terminators#.  If #ch# is a quote, strips the quotes and translates escape
 * sequences into characters.
 */
static void
ReadWord(char *toWhere,
         int ch,
         const char *terminators) {

  char *c;
  unsigned char escaped;
  int quoted;

  if((quoted = ch == '"')) {
    ch = GetCh();
    terminators = "\n\"";
  }

  for(c = toWhere;
      ch != EOF && strchr(terminators, ch) == NULL;
      ch = GetCh()) {
    if(!quoted || ch != '\\') {
      *c++ = ch;
      continue;
    }
    escaped = 0;
    ch = GetCh();
    if(ch == '0') {
      while((ch = GetCh()) >= '0' && ch <= '7')
        escaped = escaped * 8 + (ch - '0');
      UngetCh(ch);
    }
    else if(ch == 'x') {
      while(((ch = GetCh()) >= '0' && ch <= '9') ||
            (toupper(ch) >= 'A' && toupper(ch) <= 'F')) 
        escaped = escaped * 16 +
                  ((ch >= '0' && ch <= '9') ?
                   (ch - '0') : (toupper(ch) - 'A' + 10));
      UngetCh(ch);
    }
    else {
      escaped = (ch == 'a') ? '\a' :
                (ch == 'b') ? '\b' :
                (ch == 'f') ? '\f' :
                (ch == 'n') ? '\n' :
                (ch == 'r') ? '\r' :
                (ch == 't') ? '\t' :
                (ch == 'v') ? '\v' :
                ch;
    }
    *c++ = escaped;
  }

  if(ch != '"')
    UngetCh(ch);
  *c = '\0';

}


#define WHITESPACE " \f\n\r\t\v"
#define WORD_ENDERS "#;}" WHITESPACE
/*
 * Reads properties up to a closing brace and places them in #toWhere#.
 * Prepends #scopeName# to each property name.
 */
static void
ReadPropertyScope(PropertyList *toWhere,
                  const char *scopeName) {

  int ch;
  char firstWord[255 + 1];
  FILE *included;
  char name[255 + 1];
  int scopeCount = 0;
  char value[255 + 1];

  while(1) {

    /* Skip leading whitespace, property separators, and comments. */
    while(isspace((ch = GetCh())) || ch == ';' || ch == '#') {
      if(ch == '#') {
        /* See if this is a #include. */
        ReadWord(firstWord, ch, WHITESPACE);
        if(strcmp(firstWord, "#include") == 0) {
          while((ch = GetCh()) != EOF && isspace((int)ch) && ch != '\n')
            ; /* Nothing more to do. */
          if(ch != EOF && ch != '\n') {
            ReadWord(firstWord, ch, WHITESPACE);
            if((included = fopen(firstWord, "r")) != NULL)
              AddInputFile(included);
          }
        }
        else {
          while((ch = GetCh()) != EOF && ch != '\n')
            ; /* Nothing more to do. */
        }
      }
    }

    if(ch == EOF || ch == '}')
      break;

    /* Read name or first word of anonymous value and skip trailing spaces. */
    ReadWord(firstWord, ch, "=" WORD_ENDERS);
    while((ch = GetCh()) != EOF && isspace((int)ch) && ch != '\n')
      ; /* Nothing more to do. */

    if(ch == '=') {
      /* Named value; skip spaces and read first word of value, if any. */
      sprintf(name, "%s%s", scopeName, firstWord);
      while((ch = GetCh()) != EOF && isspace((int)ch) && ch != '\n')
        ; /* Nothing more to do. */
      if(ch != EOF && strchr(WORD_ENDERS, ch) == NULL) {
        ReadWord(value, ch, WORD_ENDERS);
        while((ch = GetCh()) != EOF && isspace((int)ch) && ch != '\n')
          ; /* Nothing more to do. */
      }
      else {
        strcpy(value, "");
      }
    }
    else {
      /* Anonymous value; name positionally. */
      sprintf(name, "%s%d", scopeName, scopeCount);
      strcpy(value, firstWord);
    }

    if(*value == '{') {
      /* Nested scope. */
      UngetCh(ch);
      strcat(name, ".");
      ReadPropertyScope(toWhere, name);
    }
    else {
      while(ch != EOF && strchr(WORD_ENDERS, ch) == NULL) {
        strcat(value, " ");
        ReadWord(value + strlen(value), ch, WORD_ENDERS);
        while((ch = GetCh()) != EOF && isspace((int)ch) && ch != '\n')
          ; /* Nothing more to do. */
      }
      UngetCh(ch);
      SetProperty(toWhere, name, value);
    }

    scopeCount++;

  }

}


/*
 * Writes the word #word# to #toWhere#.  Encloses the word in quotes and
 * translates any non-printing characters into escape sequences.
 */
static void
WriteWord(FILE *toWhere,
          const char *word) {
  const char *c;
  fputc('"', toWhere);
  for(c = word; *c != '\0'; c++) {
    if(*c == '"' || *c == '\\')
      fprintf(toWhere, "\\%c", *c);
    else if(*c < ' ')
      fprintf(toWhere, "\\x%X%X", *c / 16, *c % 16);
    else
      fputc(*c, toWhere);
  }
  fputc('"', toWhere);
}


/*
 * Writes to #toWhere# the elements of the array of properties pointed to by
 * #nextProperty# that belong in the scope named the #scopeNameLong# name
 * #scopeName#.  Prefixes each element with #indent# levels of indentation.
 * Updates #nextProperty# to point to the next out-of-scope element.
 */
static void
WritePropertyScope(FILE *toWhere,
                   const Property **nextProperty,
                   const char *scopeName,
                   size_t scopeNameLen,
                   size_t indent) {

  const char *equals;
  const char *name;
  const char *nameEnd;
  Property property;

  while((property = **nextProperty) != NO_PROPERTY &&
        strncmp(property, scopeName, scopeNameLen) == 0) {
    name = property + scopeNameLen;
    equals = strchr(name, '=');
    if((nameEnd = strchr(name, '.')) == NULL || nameEnd > equals)
      nameEnd = equals;
    fprintf(toWhere, "%*s%.*s = ",
            (int)indent * 2, "", (int)(nameEnd - name), name);
    if(nameEnd == equals) {
      /* Simple value. */
      WriteWord(toWhere, nameEnd + 1);
      fprintf(toWhere, "\n");
      (*nextProperty)++;
    }
    else {
      /* Compound value. */
      fprintf(toWhere, "{\n");
      WritePropertyScope
        (toWhere, nextProperty, property, nameEnd - property + 1, indent + 1);
      fprintf(toWhere, "}\n");
    }
  }

}


PropertyList
ReadPropertyList(FILE *fromWhere) {
  PropertyList returnValue = NewPropertyList();
  inputFilesCount = 0;
  AddInputFile(fromWhere);
  ReadPropertyScope(&returnValue, "");
  return returnValue;
}


void
WritePropertyList(FILE *toWhere,
                  const PropertyList list) {
  const Property *current = list;
  WritePropertyScope(toWhere, &current, "", 0, 0);
  fprintf(toWhere, "%s\n", "}");
}
