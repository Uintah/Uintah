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

/*
 * FILE: timedatafile.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 23 MAR 2005
 */
 

#ifdef _WIN32
	typedef signed __int64 int64;
	typedef unsigned __int64 uint64;

#include <stdint.h>
        typedef uint32_t u_int32_t;
#else
	typedef signed long long int64;
	typedef unsigned long long uint64;
#endif

#include <Packages/CardioWave/Core/Datatypes/TimeDataFile.h>

using namespace SCIRun;

namespace CardioWave {

TimeDataFileException::TimeDataFileException(const std::string& message)
    : message_(message)
{
}

TimeDataFileException::TimeDataFileException(const TimeDataFileException& copy)
    : message_(copy.message_)
{
}

TimeDataFileException::~TimeDataFileException()
{
}

const char* TimeDataFileException::message() const
{
    return message_.c_str();
}

const char* TimeDataFileException::type() const
{
    return "TimeDataFileException";
}


TimeDataFile::TimeDataFile(std::string filename) :
  ncols(0), nrows(0), ndims(0), elemsize(0), byteskip(0), lineskip(0)
{
  open(filename);
}

TimeDataFile::TimeDataFile() :
  ncols(0), nrows(0), ndims(0), elemsize(0), byteskip(0), lineskip(0)
{
}

TimeDataFile::~TimeDataFile()
{
}

void TimeDataFile::open(std::string filename)
{
    std::ifstream file;
    file.open(filename.c_str());
    
    ncols = 0;
    nrows = 0;
    byteskip = 0;
    lineskip = 0;

    std::string line;
     
    while(!file.eof())
    { 
      getline(file,line);
      std::string::size_type colon = line.find(":");
      if (colon < (line.size()-1))
      {
        std::string keyword = remspaces(line.substr(0,colon));
        std::string attribute = line.substr(colon+1);
        
        if (cmp_nocase(keyword,"encoding") == 0) encoding = remspaces(attribute);
        if (cmp_nocase(keyword,"type") == 0) type = remspaces(attribute);
        if (cmp_nocase(keyword,"datafile") == 0) datafilename = remspaces(attribute);
        if (cmp_nocase(keyword,"content") == 0) content = attribute;
        if (cmp_nocase(keyword,"sampleunits") == 0) unit = remspaces(attribute);
        if (cmp_nocase(keyword,"endian") == 0) endian = remspaces(attribute);
        if (cmp_nocase(keyword,"dimension")  == 0)
        { 
          std::istringstream iss(attribute);
          iss >> dimension;
        }
        if (cmp_nocase(keyword,"lineskip") == 0)
        { 
          std::istringstream iss(attribute);
          iss >> dimension;
        }

        if (cmp_nocase(keyword,"sizes") == 0)
        { 
          std::istringstream iss(attribute);
          iss >> nrows;
          iss >> ncols;
        }
      }
    }
  
    // We only support one of Gordon's types
    // If some wants more it would be easier to fix teem library
    if ((encoding != "raw")&&(encoding!=""))
    {
      throw TimeDataFileException("Encoding must be raw");
    }
    encoding = "raw";
    
    gettype(type,ntype,elemsize);
    if (ntype == 0)
    {
      throw TimeDataFileException("Unknown encoding encounterd");    
    }

    if (datafilename == "")
    {
      throw TimeDataFileException("No data file specified, separate headers are required");    
    }

    if ((endian != "big")&&(endian != "little")&&(endian != ""))
    {
      throw TimeDataFileException("Unknown endian type encountered");  
    }
    
    swapbytes = false;
    short test = 0x00FF;
    char *testptr = reinterpret_cast<char *>(&test);
    if ((testptr[0])&&(endian == "little")) swapbytes = true;
    if ((testptr[1])&&(endian == "big")) swapbytes = true;

    if ((nrows < 1)||(ncols < 1))
    {
      throw TimeDataFileException("Improper NRRD dimensions: number of columns/rows is smaller then one");  
    }
    
    datafile = fopen(datafilename.c_str(),"r");
    if (datafile == 0)
    {
        std::string::size_type slash = filename.size();
        std::string fn = filename;
        slash = fn.rfind("/");
        if (slash  < filename.size())
        {
          datafilename = filename.substr(0,slash+1) + datafilename;
        }
        datafile = fopen(datafilename.c_str(),"r");
        if (datafile == 0)
        {
          throw TimeDataFileException("Could not find/open datafile");    
        }
    }
    fclose(datafile);
}

int TimeDataFile::getncols()
{
  return(ncols);
}

int TimeDataFile::getnrows()
{
  return(nrows);
}
    
std::string TimeDataFile::getcontent()
{
  return(content);
}

std::string TimeDataFile::getunit()
{
  return(unit);
}

void TimeDataFile::getcolmatrix(SCIRun::MatrixHandle& mh,int colstart,int colend)
{

  if (colstart > colend)
  {
    throw TimeDataFileException("Column start is bigger than column end");
  }

  datafile = fopen(datafilename.c_str(),"rb");
  if (datafile == 0) 
  {
    throw TimeDataFileException("Could not find/open datafile");
  }

  if (lineskip > 0)
  {
     char buffer;
     int ln = lineskip;
     while (ln)
     {
        if(fread(&buffer,1,1,datafile) != 1)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read header of datafile"); 
        }
        if (buffer == '\n') ln--;
     }
  }
  
  // Accoring to Gordon's definition
  if (byteskip >= 0)
  {
    if (fseek(datafile,byteskip,SEEK_CUR)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  else
  {
    if (fseek(datafile,ncols*nrows*elemsize,SEEK_END)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  
  int numcols = colend-colstart + 1;
  
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(nrows,numcols);
  
  if (mat == 0)
  {
      fclose(datafile);
      throw TimeDataFileException("Could not allocate matrix");
  }  

  void* buffer = reinterpret_cast<void *>(mat->getData());
  
  if (fseek(datafile,elemsize*nrows*colstart,SEEK_CUR)!=0)
  {
    fclose(datafile);
    delete mat;
    throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
  }    


  if (numcols*nrows != fread(buffer,elemsize,numcols*nrows,datafile))
  {
    fclose(datafile);
    delete mat;
  }
  if (swapbytes) doswapbytes(buffer,elemsize,numcols*nrows);

  if (ntype == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*nrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]);}

  fclose(datafile);
  mh = dynamic_cast<Matrix *>(mat->transpose());
  delete mat;
}

void TimeDataFile::getcolnrrd(NrrdDataHandle& mh,int colstart,int colend)
{

  if (colstart > colend)
  {
    throw TimeDataFileException("Column start is bigger than column end");
  }

  datafile = fopen(datafilename.c_str(),"rb");
  if (datafile == 0) 
  {
    throw TimeDataFileException("Could not find/open datafile");
  }

  if (lineskip > 0)
  {
     char buffer;
     int ln = lineskip;
     while (ln)
     {
        if(fread(&buffer,1,1,datafile) != 1)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read header of datafile"); 
        }
        if (buffer == '\n') ln--;
     }
  }
  
  // Accoring to Gordon's definition
  if (byteskip >= 0)
  {
    if (fseek(datafile,byteskip,SEEK_CUR)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  else
  {
    if (fseek(datafile,ncols*nrows*elemsize,SEEK_END)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  
  int numcols = colend-colstart + 1;
  
  SCIRun::NrrdData *nrrd = scinew SCIRun::NrrdData();
  if (nrrd == 0)
  {
      fclose(datafile);
      throw TimeDataFileException("Could not allocate nrrd object");
  }  

  nrrd->nrrd = nrrdNew();
  if (nrrd->nrrd == 0)
  {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Could not allocate nrrd");
  }  

  nrrdAlloc(nrrd->nrrd,ntype,2,nrows,numcols);
  if (nrrd->nrrd->data == 0)
  {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Could not allocate nrrd");
  }  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoSpacing,1.0,1.0);
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMin,0.0,0.0);  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMax,static_cast<double>(nrows),static_cast<double>(numcols));
  

  void* buffer = reinterpret_cast<void *>(nrrd->nrrd->data);
  
  if (fseek(datafile,elemsize*nrows*colstart,SEEK_CUR)!=0)
  {
    fclose(datafile);
    delete nrrd;
    throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
  }    


  if (numcols*nrows != fread(buffer,elemsize,numcols*nrows,datafile))
  {
    fclose(datafile);
    delete nrrd;
    throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
  }
  if (swapbytes) doswapbytes(buffer,elemsize,numcols*nrows);

  fclose(datafile);
  mh = nrrd;
}


void TimeDataFile::getrownrrd(SCIRun::NrrdDataHandle& mh,int rowstart,int rowend)
{

  if (rowstart > rowend)
  {
    throw TimeDataFileException("Column start is bigger than column end");
  }

  datafile = fopen(datafilename.c_str(),"rb");
  if (datafile == 0) 
  {
    throw TimeDataFileException("Could not find/open datafile");
  }

  if (lineskip > 0)
  {
     char buffer;
     int ln = lineskip;
     while (ln)
     {
        if(fread(&buffer,1,1,datafile) != 1)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read header of datafile"); 
        }
        if (buffer == '\n') ln--;
     }
  }
  
  // Accoring to Gordon's definition
  if (byteskip >= 0)
  {
    if (fseek(datafile,byteskip,SEEK_CUR)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  else
  {
    if (fseek(datafile,ncols*nrows*elemsize,SEEK_END)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  
  int numrows = rowend-rowstart + 1;
  
  SCIRun::NrrdData *nrrd = scinew SCIRun::NrrdData();
  if (nrrd == 0)
  {
      fclose(datafile);
      throw TimeDataFileException("Could not allocate nrrd object");
  }  

  nrrd->nrrd = nrrdNew();
  if (nrrd->nrrd == 0)
  {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Could not allocate nrrd");
  }  

  nrrdAlloc(nrrd->nrrd,ntype,2,numrows,ncols);
  if (nrrd->nrrd->data == 0)
  {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Could not allocate nrrd");
  }  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoSpacing,1.0,1.0);
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMin,0.0,0.0);  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMax,static_cast<double>(numrows),static_cast<double>(ncols));
  
  char* buffer = reinterpret_cast<char *>(nrrd->nrrd->data);
   
  if (fseek(datafile,elemsize*rowstart,SEEK_CUR)!=0)
  {
    fclose(datafile);
    delete nrrd;
    throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
  }    

  for (int i=0;i<ncols;i++)
  {
    if (numrows != fread(buffer+(elemsize*i*numrows),elemsize,numrows,datafile))
    {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
    }
    if (fseek(datafile,elemsize*(nrows-numrows),SEEK_CUR)!=0)
    {
      fclose(datafile);
      delete nrrd;
      throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
    }    
  }
    
  if (swapbytes) doswapbytes(buffer,elemsize,numrows*ncols);

  fclose(datafile);
  mh = nrrd;
}



void TimeDataFile::getrowmatrix(SCIRun::MatrixHandle& mh,int rowstart,int rowend)
{

  if (rowstart > rowend)
  {
    throw TimeDataFileException("Column start is bigger than column end");
  }

  datafile = fopen(datafilename.c_str(),"rb");
  if (datafile == 0) 
  {
    throw TimeDataFileException("Could not find/open datafile");
  }

  if (lineskip > 0)
  {
     char buffer;
     int ln = lineskip;
     while (ln)
     {
        if(fread(&buffer,1,1,datafile) != 1)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read header of datafile"); 
        }
        if (buffer == '\n') ln--;
     }
  }
  
  // Accoring to Gordon's definition
  if (byteskip >= 0)
  {
    if (fseek(datafile,byteskip,SEEK_CUR)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  else
  {
    if (fseek(datafile,ncols*nrows*elemsize,SEEK_END)!=0)
    {
      fclose(datafile);
      throw TimeDataFileException("Could not read datafile");
    }
  }
  
  int numrows =rowend-rowstart + 1;
  
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(numrows,ncols);
  
  if (mat == 0)
  {
      fclose(datafile);
      throw TimeDataFileException("Could not allocate matrix");
  }  

  char* buffer = reinterpret_cast<char *>(mat->getData());
  
  if (fseek(datafile,elemsize*rowstart,SEEK_CUR)!=0)
  {
    fclose(datafile);
    delete mat;
    throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
  }    

  for (int i=0;i<ncols;i++)
  {
    if (numrows != fread(buffer+(elemsize*i*numrows),elemsize,numrows,datafile))
    {
      fclose(datafile);
      delete mat;
      throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
    }
    if (fseek(datafile,elemsize*(nrows-numrows),SEEK_CUR)!=0)
    {
      fclose(datafile);
      delete mat;
      throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
    }    
  }
  if (swapbytes) doswapbytes(buffer,elemsize,numrows*ncols);

  if (ntype == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(ncols*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  
  fclose(datafile);
  mh = dynamic_cast<Matrix *>(mat->transpose());
  delete mat;
}



void TimeDataFile::gettype(std::string type,int& nrrdtype, int& elsize)
{
    if ((type == "char")||(type == "signedchar")||(type == "int8")||(type == "int8_t"))
    {
        nrrdtype = nrrdTypeChar; elsize = 1; return;
    }
    if  ((type == "unsignedchar")||(type == "uchar")||(type == "unit8")||(type == "uint8_t"))
    {
      nrrdtype = nrrdTypeUChar; elsize= 1; return;
    }
    if ((type == "short")||(type == "shortint")||(type == "signedshort")||(type == "signedshortint")||(type == "int16")||(type == "int_16"))
    {
        nrrdtype = nrrdTypeShort; elsize = 2; return;
    }
    if  ((type == "ushort")||(type == "unsignedshort")||(type == "unsignedshortint")||(type == "uint16")||(type == "uint16_t"))
    {
      nrrdtype = nrrdTypeUShort; elsize= 2; return;
    }
    if ((type == "int")||(type == "signedint")||(type == "int32")||(type == "int32_t")||(type == "long")||(type == "longint"))
    {
        nrrdtype = nrrdTypeInt; elsize = 4; return;
    }
    if  ((type == "uint")||(type == "unsignedint")||(type == "uint32")||(type == "uint32_t")||(type == "unsignedlong")||(type == "unsignedlongint"))
    {
      nrrdtype = nrrdTypeUInt; elsize= 4; return;
    }  
    if ((type == "longlong")||(type == "longlongint")||(type == "signedlonglong")||(type == "signedlonglongint")||(type == "int64")||(type == "int64_t"))
    {
        nrrdtype = nrrdTypeLLong; elsize = 8; return;
    }
    if  ((type == "unsignedlonglong")||(type == "unsignedlonglongint")||(type == "uint64")||(type == "uint64_t")||(type == "ulonglong"))
    {
      nrrdtype = nrrdTypeULLong; elsize= 8; return;
    }  
    if  ((type == "float")||(type == "single"))
    {
      nrrdtype = nrrdTypeFloat; elsize= 4; return;
    }        
    if  ((type == "double"))
    {
      nrrdtype = nrrdTypeDouble; elsize= 8; return;
    }        
        
    nrrdtype = 0;
    elsize = 0;    
}


int TimeDataFile::cmp_nocase(const std::string& s1, const std::string& s2)
{
  std::string::const_iterator p1 = s1.begin();
  std::string::const_iterator p2 = s2.begin();
  
  while (p1 != s1.end() && p2 != s2.end())
  {
    if (toupper(*p1) != toupper(*p2)) return ( toupper(*p1) < toupper(*p2)) ? -1 : 1;
    ++p1;
    ++p2;
  }
  
  return(s2.size() == s1.size()) ? 0 : (s1.size() < s2.size()) ? -1 : 1;
}

std::string TimeDataFile::remspaces(std::string str)
{
  std::string newstr;
  for (std::string::size_type i=0;i<str.size();i++) if ((str[i] != ' ' )&&(str[i] != '\t')) newstr += str[i];
  return(newstr);
}

void TimeDataFile::doswapbytes(void *vbuffer,long elsize,long size)
{
   char temp;
   char *buffer = static_cast<char *>(vbuffer);

   size *= elsize;

   switch(elsize)
   {
      case 0:
      case 1:
         // Do nothing. Element size is 1 byte, so there is nothing to swap
         break;
      case 2:  
		// Do a 2 bytes element byte swap. 
		for(long p=0;p<size;p+=2)
		  { temp = buffer[p]; buffer[p] = buffer[p+1]; buffer[p+1] = temp; }
		break;
      case 4:
		// Do a 4 bytes element byte swap.
		for(long p=0;p<size;p+=4)
		  { temp = buffer[p]; buffer[p] = buffer[p+3]; buffer[p+3] = temp; 
			temp = buffer[p+1]; buffer[p+1] = buffer[p+2]; buffer[p+2] = temp; }
		break;
      case 8:
		// Do a 8 bytes element byte swap.
		for(long p=0;p<size;p+=8)
		  { temp = buffer[p]; buffer[p] = buffer[p+7]; buffer[p+7] = temp; 
			temp = buffer[p+1]; buffer[p+1] = buffer[p+6]; buffer[p+6] = temp; 
			temp = buffer[p+2]; buffer[p+2] = buffer[p+5]; buffer[p+5] = temp; 
			temp = buffer[p+3]; buffer[p+3] = buffer[p+4]; buffer[p+4] = temp; }
   	    break;
      default:
       throw TimeDataFileException("Unknown element size encounterd");    
   }  
}

}

