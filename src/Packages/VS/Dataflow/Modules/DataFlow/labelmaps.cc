/******************************************************************************
 * File: labelmaps.cc
 *
 * Description: C source code for classes to provide an API for Visible Human
 *		segmentation information:  The Master Anatomy Label Map
 *		Spatial Adjacency relations for the anatomical structures
 *		and Bounding BOxes for each anatmocal entity.
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include "labelmaps.h"

using namespace std;

char *read_line(char *retbuff, int *bufsize, FILE *infile)
{
char inbuff[VH_FILE_MAXLINE];
int readsize = VH_FILE_MAXLINE;

// read a MAXLINE-byte chunk of the file
char *stat = fgets(inbuff, readsize, infile);
                                                                                
if(stat)
  {
    int readlen = strlen(stat);
    int bufflen = strlen(retbuff);
    bool longline = (readlen == VH_FILE_MAXLINE-1) &&
                         (stat[readlen-1] != '\n');
    bool continued = false;
    if(readlen > 1 && stat[readlen-2] == '\\')
      { // reading continued lines, indicated by '<backslash>-<newline>'
        continued = true;
        readlen -= 2;
      }
    if(longline || continued)
      { // reading a line longer than MAXLINE
        int cumlen = bufflen + readlen + 1;
                                                                                
//      cerr << "\nread_line: ";
                                                                                
        if(cumlen > *bufsize)
          {
//          cerr << "read " << readlen;
//          cerr << " chars into buffer[" << *bufsize << "]";
                                                                                
            // allocate a new buffer for returned string
            *bufsize = *bufsize * 2;
//          retbuff = (char *)realloc( retbuff, *bufsize );
            char *newBuf = new char[*bufsize];
            bcopy(retbuff, newBuf, bufflen+1);
            delete [] retbuff;
            retbuff = newBuf;
            // with new buffsize
          }
//      cerr << "retbuff '" << retbuff << "' inbuff '" << inbuff << "'" << endl;                                                                                
        // append contents of read to return buffer
        strncat(retbuff, inbuff, readlen);
                                                                                
        // continue reading
//      cerr << " read_line calling read_line" << endl;
        retbuff = read_line(retbuff, bufsize, infile);
      }
    else
      { // the total length of the line is less than bufsize
        strcat(retbuff, inbuff);
      }
    return(retbuff);
  }
else
  {
    return((char *)0);
  }
} // end read_line()

VH_MasterAnatomy::VH_MasterAnatomy()
{
  anatomyname = new (char *)[VH_LM_NUM_NAMES];
  labelindex = new int[VH_LM_NUM_NAMES];
  num_names = 0;
}

VH_MasterAnatomy::~VH_MasterAnatomy()
{
  delete [] anatomyname;
  delete [] labelindex;
}

void
VH_MasterAnatomy::readFile(FILE *infileptr)
{
} // end VH_MasterAnatomy::readFile(FILE *infileptr)

int
splitAtComma(char *srcStr, char **dst0, char **dst1)
{
  char *srcPtr = srcStr;
  int count = 0;

  if(!srcStr || !dst0 || !dst1)
  {
    cerr << "VH_MasterAnatomy::splitAtComma(): NULL string" << endl;
    return(0);
  }
  while(*srcPtr != ',') { srcPtr++; count++; }
#ifdef __APPLE__
  *dst0 = (char *)malloc((count + 1) * sizeof(char));
  bzero(*dst0, count + 1);
  strncpy(*dst0, srcStr, count);
#else
  *dst0 = strndup(srcStr, count);
#endif
  // srcPtr points to ','
  *dst1 = ++srcPtr;
  return(1);
}

void
VH_MasterAnatomy::readFile(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_MasterAnatomy::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;

  cerr << "VH_MasterAnatomy::readFile(" << infilename << ")";
  char *indexStr;
  // skip first line
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_MasterAnatomy::readFile(): premature EOF" << endl;
  }
  // label 0 is "unknown"
  anatomyname[num_names] = strdup("unknown");
  labelindex[num_names] = 0;
  num_names++;
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form: AnatomyName,Label
      if(!splitAtComma((char *)inLine,
                   &anatomyname[num_names], &indexStr)) break;
      labelindex[num_names] = atoi(indexStr);
    } // end if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
    num_names++;
    cerr << ".";
  } // end while(read_line(inLine, &buffsize, infile) != 0)

  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
} // end VH_MasterAnatomy::readFile(char *infilename)

char *
VH_MasterAnatomy::get_anatomyname(int labelindex)
{
  return(anatomyname[labelindex]);
} // end VH_MasterAnatomy::get_anatomyname(int labelindex)

int
VH_MasterAnatomy::get_labelindex(char *targetname)
{
  for(int index = 0; index < num_names; index++)
  {
    if(anatomyname[index] != 0 && strcmp(anatomyname[index], targetname) == 0)
      return(index);
  }

  return(-1);
} // end VH_MasterAnatomy::get_labelindex(char *anatomyname)

VH_AdjacencyMapping::VH_AdjacencyMapping()
{
  rellist = new (int *)[VH_LM_NUM_NAMES];
  numrel = new int[VH_LM_NUM_NAMES];
  num_names = 0;
}

VH_AdjacencyMapping::~VH_AdjacencyMapping()
{
  delete [] rellist;
  delete [] numrel;
}
int
countCommas(char *inString)
{
  int count = 0;
  while(*inString != '\0')
  {
    if(*inString++ == ',') count++;
  }
  return count;
}

void
VH_AdjacencyMapping::readFile(char *infilename)
{
  FILE *infile;
  char *inLine;
  if(!(infile = fopen(infilename, "r")))
  {
    perror("VH_AdjacencyMapping::readFile()");
    cerr << "cannot open '" << infilename << "'" << endl;
    return;
  }

  inLine = new char[VH_FILE_MAXLINE];
  int buffsize = VH_FILE_MAXLINE;

  cerr << "VH_AdjacencyMapping::readFile(" << infilename << ")";
  num_names = 0;
  char *indexStr;
  // skip the first line
  if((inLine = read_line(inLine, &buffsize, infile)) <= 0)
  {
    cerr << "VH_AdjacencyMapping::readFile(): premature EOF" << endl;
  }
  while(read_line(inLine, &buffsize, infile) != 0)
  {
    if(strlen(inLine) > 0)
    { // expect lines of the form: index0,index1,...indexn
      int num_adj = countCommas(inLine);
      numrel[num_names] = num_adj;
      // cerr << "line[" << num_names << "], " << strlen(inLine) << " chars, ";
      // cerr << num_adj << " relations" << endl;
      rellist[num_names] = new int[num_adj];
      indexStr = strtok(inLine, ",");
      // first index is the MasterAnatomy name index
      if(atoi(indexStr) != num_names)
      {
        cerr << "VH_AdjacencyMapping::readFile(): line index mismatch: ";
        cerr << indexStr << "(" << atoi(indexStr) << ") <=> " << num_names;
        cerr << endl;
      }
      int *intPtr = rellist[num_names];
      while((indexStr = strtok(NULL, ",")) != NULL)
      {
        *intPtr++ = atoi(indexStr);
      }
      num_names++;
      cerr << ".";
    } if(strlen(inLine) > 0)
    // (else) blank line -- ignore
    // clear input buffer line
    strcpy(inLine, "");
  } // end while(read_line(inLine, &buffsize, infile) != 0)
  delete [] inLine;
  fclose(infile);
  cerr << "done" << endl;
} // end VH_AdjacencyMapping::readFile(char *infilename)

void
VH_AdjacencyMapping::readFile(FILE *infileptr)
{
} // end VH_AdjacencyMapping::readFile(FILE *infileptr)

int *
VH_AdjacencyMapping::adjacent_to(int index)
{
  if(index < num_names)
     return rellist[index];
  else
  {
     cerr << "VH_AdjacencyMapping::adjacent_to(): index " << index;
     cerr << " out of range " << num_names << endl;
  }
  // return the relation list for "unknown"
  return(rellist[0]);
}

int 
VH_AdjacencyMapping::get_num_rel(int index)
{
  if(index < num_names)
     return numrel[index];
  else
  {
     cerr << "VH_AdjacencyMapping:::get_num_rel(): index " << index;
     cerr << " out of range " << num_names << endl;
  }
  // return the number of relations for "unknown"
  return(numrel[0]);
}

void
VH_AnatomyBoundingBox::readFile(char *infilename)
{
} // end VH_AnatomyBoundingBox::readFile(char *infilename)

void
VH_AnatomyBoundingBox::readFile(FILE *infileptr)
{
} // end VH_AnatomyBoundingBox::readFile(FILE *infileptr)


