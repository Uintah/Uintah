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
  
  #define __USE_LARGEFILE64
  #include <fcntl.h>

  #ifndef O_LARGEFILE
  #define O_LARGEFILE 0
  #endif

	typedef signed long long int64;
	typedef unsigned long long uint64;
#endif

#include <Packages/CardioWave/Core/Datatypes/TimeDataFile.h>

using namespace SCIRun;
using namespace std;

#include <sys/stat.h>
#include <Core/OS/Dir.h>  // for LSTAT

#include <sci_defs/config_defs.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#include <fcntl.h>
#endif

#include <stdio.h>
#include <ctype.h>


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
  byteskip_(0), lineskip_(0), elemsize_(0), ntype_(0), dimension_(0),
  start_(0), end_(0), step_(0), subdim_(0), useformatting_(false), swapbytes_(false)
{
  open(filename);
}

TimeDataFile::TimeDataFile() :
  byteskip_(0), lineskip_(0), elemsize_(0), ntype_(0), dimension_(0),
  start_(0), end_(0), step_(0), subdim_(0), useformatting_(false), swapbytes_(false)

{
}

TimeDataFile::~TimeDataFile()
{
}

int TimeDataFile::getsize(int dim)
{
  if (dim >= sizes_.size()) return (0);  
  return (sizes_[dim]);
}

double TimeDataFile::getspacing(int dim)
{
  return (spacings_[dim]);	
}

std::string TimeDataFile::getunit(int dim)
{
  return (units_[dim]);
}

std::string TimeDataFile::getkind(int dim)
{
  return (kinds_[dim]);
}


void TimeDataFile::open(std::string filename)
{
    std::ifstream file;
    
    int ntries = 0;
    bool success = false;
    
    // Since the header file maybe being updated
    // This way the reader will try several times during 0.2
    // seconds to get access to it.
    // It is not the best synchronization, unfortunately the
    // nrrd format is ill designed and does not allow for
    // dynamically adding files by using only a template of a
    // file. Hence the use of this hack.
    
    while ((ntries < 10)&&(success == false))
    {
      try
      {
        file.open(filename.c_str());
        success = true;
        ntries++;
      }
      catch(...)
      {
        SCIRun::Time::waitFor(0.2);
      }
    }
      
    byteskip_ = 0;
    lineskip_ = 0;
    keyvalue_.clear();
    datafilename_ = "";
    datafilenames_.clear();
    units_.clear();
    kinds_.clear();
    spacings_.clear();
    sizes_.clear();
    coloffset_.clear();
    content_ = "";
    endian_ = "";
    type_ = "";
    unit_ = "";
    dimension_ = 0;
    ntype_ = 0;
    elemsize_ = 0;
    subdim_ = 0;
    start_ = 0;
    end_ = 0;
    step_ = 0;
    useformatting_ = false;

    std::string line;
    
    int numlines = 0; 
       
    while(!file.eof())
    { 
      getline(file,line);
      numlines++;
      
      if ((line.size() == 0)&&(datafilename_ == ""))
      {
        // Apparently this is a combined header data file;
        // Stop reading header and telll reader how many lines
        // to skip.
        lineskip_ += numlines;
        datafilename_ = filename;
        break;
      }
      
      std::string::size_type colon = line.find(":");
      if (colon < (line.size()-1))
      {
        std::string keyword = remspaces(line.substr(0,colon));
        std::string attribute = line.substr(colon+1);
        
        if ((attribute.size() > 0)&&(attribute[0] == '='))
        {
          // we have a key value pair.
          std::string value = line.substr(colon+2);
          keyvalue_[keyword] = value;
          continue;
        }
        
        if (cmp_nocase(keyword,"encoding") == 0) encoding_ = remspaces(attribute);
        if (cmp_nocase(keyword,"type") == 0) type_ = remspaces(attribute);
        if (cmp_nocase(keyword,"units")== 0)
        {
          std::string::size_type startstr = attribute.find('"');
          while (startstr+1 < attribute.size())
          {
            attribute = attribute.substr(startstr+1);
            std::string::size_type endstr = attribute.find('"');
            if (endstr >= attribute.size()) break;
            std::string unitstr = attribute.substr(0,endstr);
            units_.push_back(unitstr);
            attribute = attribute.substr(endstr+1);
            startstr = attribute.find('"');
          }
        }

        if (cmp_nocase(keyword,"kinds")== 0)
        {
          std::string::size_type startstr = attribute.find('"');
          while (startstr+1 < attribute.size())
          {
            attribute = attribute.substr(startstr+1);
            std::string::size_type endstr = attribute.find('"');
            if (endstr >= attribute.size()) break;
            std::string kindstr = attribute.substr(0,endstr);
            kinds_.push_back(kindstr);
            attribute = attribute.substr(endstr+1);
            startstr = attribute.find('"');
          }
        }
        
        
        if (cmp_nocase(keyword,"spacings")== 0)
        {
          std::istringstream iss(attribute);
          iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
          try
          {
            while(1)
            {
              double spacing;
              iss >> spacing;
              spacings_.push_back(spacing);
            }
          }
          catch(...)
          {
          }
        }

        if (cmp_nocase(keyword,"datafile") == 0) 
        {
          std::string::size_type percent = attribute.find('%');
          if (percent < attribute.size())
          {
             std::istringstream iss(attribute); 
             iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
             start_ = 1;
             end_ = 1;
             step_ = 1;
             subdim_ = 0;
             useformatting_ = true;
             
             try
             {
              iss >> datafilename_;
              iss >> start_;
              iss >> end_;
              iss >> step_;
              iss >> subdim_;
             }
             catch(...)
             {
             }
             
          }
          else
          {
              datafilename_ = remspaces(attribute);
          }

          if (datafilename_.size() >= 4)
          {
            if (datafilename_.substr(0,4) == "LIST")
            {
              if(datafilename_.size() > 4)
              {
                std::istringstream iss(datafilename_.substr(4));
                subdim_ = 0;
                iss >> subdim_;
              }
              while(!file.eof())
              { 
                getline(file,line);
                numlines++;
                datafilenames_.push_back(remspaces(line));
              }
            }
          }
        }
        if (cmp_nocase(keyword,"content") == 0) content_ = attribute;
        if (cmp_nocase(keyword,"sampleunits") == 0) unit_ = remspaces(attribute);
        if (cmp_nocase(keyword,"endian") == 0) endian_ = remspaces(attribute);
        if (cmp_nocase(keyword,"dimension")  == 0)
        { 
          std::istringstream iss(attribute);
          iss >> dimension_;
        }
        if (cmp_nocase(keyword,"lineskip") == 0)
        { 
          std::istringstream iss(attribute);
          iss >> lineskip_;
        }

        if (cmp_nocase(keyword,"sizes") == 0)
        { 
          std::istringstream iss(attribute);
          iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
          try
          {
            while(1)
            {
              int size;
              iss >> size;
              sizes_.push_back(size);
            }
          }
          catch(...)
          {
          }
        }
      }
    }
	
	  if (dimension_ == 0) dimension_ = sizes_.size();
	
    // Correct the subdim values
    if ((subdim_ > dimension_)&&(subdim_ < 0))
    {
      throw TimeDataFileException("Improper subdim value encountered");    
    }
    if (subdim_ == 0) subdim_ = 1;
    
    if (subdim_ == dimension_) subdim_ = dimension_-1;
    
    if (dimension_ > sizes_.size())
    {
      throw TimeDataFileException("The size of sizes array does not match dimension in NRRD file");
    }
    
    // complete the per axis information
    for (size_t p = spacings_.size(); p < dimension_; p++ ) spacings_.push_back(1.0);
    for (size_t p = units_.size(); p < dimension_; p++ ) units_.push_back("");
    for (size_t p = kinds_.size(); p < dimension_; p++ ) kinds_.push_back("");
    for (size_t p = sizes_.size(); p < dimension_; p++ ) sizes_.push_back(1);
  
    // We only support one of Gordon's types
    // If someone wants more it would be easier to fix teem library
    if ((encoding_ != "raw")&&(encoding_!=""))
    {
      throw TimeDataFileException("Encoding must be raw");
    }
    encoding_ = "raw";
    
    gettype(type_,ntype_,elemsize_);
    if (ntype_ == 0)
    {
      throw TimeDataFileException("Unknown encoding encounterd");    
    }

    FILE *datafile;

    if (useformatting_)
    {
      char *buffer = scinew char[datafilename_.size()+40];
      std::string  newfilename;
      datafilenames_.clear();
      
      bool foundend = false;
      
      for (int p=start_;((p<=end_)||(end_ == -1))&&(!foundend);p+=step_)
      {
        ::snprintf(&(buffer[0]),datafilename_.size()+39,datafilename_.c_str(),p);
        buffer[datafilename_.size()+39] = 0;
        newfilename = buffer;

        datafile = fopen(newfilename.c_str(),"rb");

        if (datafile == 0)
        {
          std::string::size_type slash = filename.size();
          std::string fn = filename;
          slash = fn.rfind("/");
          if (slash  < filename.size())
          {
            newfilename = filename.substr(0,slash+1) + newfilename;
          }
               
          datafile = fopen(newfilename.c_str(),"rb");
          if (datafile == 0)
          {
                 
            if ((ncols_ == -1)||(end_ == -1))
            {
              foundend = true;
            }
            else
            {
              throw TimeDataFileException("Could not find/open datafile: "+newfilename);    
            }
          }
          else
          {
            fclose(datafile);
            datafilenames_.push_back(newfilename);
          }
        }
        else
        {
          fclose(datafile);
          datafilenames_.push_back(newfilename);
        }
      }
      delete[] buffer;
    }
   
    if (datafilename_ == "")
    {
      throw TimeDataFileException("No data file specified, separate headers are required");    
    }

    if ((endian_ != "big")&&(endian_ != "little")&&(endian_ != ""))
    {
      throw TimeDataFileException("Unknown endian type encountered");  
    }
    
    swapbytes_ = false;
    short test = 0x00FF;
    char *testptr = reinterpret_cast<char *>(&test);
    if ((testptr[1])&&(endian_ == "little")) swapbytes_ = true;
    if ((testptr[0])&&(endian_ == "big")) swapbytes_ = true;


	for (size_t p =0; p < (dimension_-1) ; p++)
	{
		if (sizes_[p] <  1) throw TimeDataFileException("Improper NRRD dimensions: number of columns/rows is smaller then one");  
	}
	
	if (sizes_[dimension_-2] < -1)
	{
		throw TimeDataFileException("Improper NRRD dimensions: number of columns/rows is smaller then one");
	}
	
    if (datafilenames_.size() == 0)
    {
      datafile = fopen(datafilename_.c_str(),"rb");
      if (datafile == 0)
      {
          std::string::size_type slash = filename.size();
          std::string fn = filename;
          slash = fn.rfind("/");
          if (slash  < filename.size())
          {
            datafilename_ = filename.substr(0,slash+1) + datafilename_;
          }
          
          datafile = fopen(datafilename_.c_str(),"rb");
          if (datafile == 0)
          {
            throw TimeDataFileException("Could not find/open datafile: "+datafilename_);    
          }
      }

      fclose(datafile);    
    }  

    int nsd =0;
    if (datafilenames_.size() == 0)
    {
      struct stat buf;
      if (LSTAT(datafilename_.c_str(),&buf) < 0)
      {
        throw TimeDataFileException("Could not determine size of datafile");          
      }
      ncolsr = static_cast<int>((buf.st_size)/static_cast<off_t>(nrows_*elemsize_));
    }
    else
    {
      nsd = 0;
      std::list<std::string>::iterator p = datafilenames_.begin();
      int q = 0;
      sdsize_ = 1;
      for (size_p =0; p < subdim_; p++) sdsize_ *= sizes_[p]; 
	  
      sdoffset_.resize(datafilenames_.size()+1);
      for (;p != datafilenames_.end();p++)
      {
        struct stat buf;
        if (LSTAT((*p).c_str(),&buf) < 0)
        {
            throw TimeDataFileException("Could not determine size of datafile");          
        }
        sdoffset_[q++] = nsd;
        nsd += static_cast<int>((buf.st_size)/static_cast<off_t>(sdsize_*elemsize_));
      }
      sdoffset_[q] = nsd;
    }

    int div = 1;
    for (size_t p = subdim_ ; p < dimension_-1; p++ ) div *= sizes_[p]; 
    int lastdim = static_cast<int>(nsd/div);
      
    if (sizes_[dimension_-1] == -1) sizes_[dimension_-1] = lastdim; 
    if (sizes_[dimension-1] > lastdim) sizes_[dimension_-1] = lastdim;
    
    if (datafilenames_.size() > 0)
    {
      if (byteskip_ != 0)  throw TimeDataFileException("Byteskip and data spread out over multiple files is not supported yet");
      if (lineskip_ != 0)  throw TimeDataFileException("Lineskip and data spread out over multiple files is not supported yet");
    }
}

int TimeDataFile::getncols()
{
  return(sizes_[1]);
}

int TimeDataFile::getnrows()
{
  return(sizes_[0]);
}
    
std::string TimeDataFile::getcontent()
{
  return(content_);
}

std::string TimeDataFile::getsampleunit()
{
  return(unit_);
}


void TimeDataFile::getcolmatrix(SCIRun::MatrixHandle& mh,int colstart,int colend)
{
  FILE*                   datafile;
  int                     datafile_uni;
 
  if (dimension_ > 2) throw TimeDataFileException("Matrix format only supports systems of 2 dimensions");

  if (colstart > colend) throw TimeDataFileException("Column start is bigger than column end");

  int numcols = colend-colstart + 1;      
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(numcols,nrows_);
  if (mat == 0) throw TimeDataFileException("Could not allocate matrix");  
  SCIRun::MatrixHandle handle = dynamic_cast<SCIRun::Matrix *>(mat);

  char* buffer = reinterpret_cast<char *>(mat->getData());

  int c = colstart;
  while(c<=colend)
  {

    int coffset = c;
    int colread = (colend-c)+1;
    std::string fn = datafilename_;
    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (p=0;p<sdoffset_.size();p++) { if((c >= sdoffset_[p] )&&(c < sdoffset_[p+1])) break;}
      if (p == sdoffset_.size()) throw TimeDataFileException("Column index out of range");
      coffset = c-sdoffset_[p];
      if (colend < sdoffset_[p+1]) colread = (colend-c)+1; else colread = (sdoffset_[p+1]-c);
      std::list<std::string>::iterator it = datafilenames_.begin();
      for (int q=0;q<p;q++) it++;
      fn = (*it);
    }
    
    c+=colread;
    
    #ifndef HAVE_UNISTD_H 
    
      datafile = fopen(fn.c_str(),"rb");
      if (datafile == 0) throw TimeDataFileException("Could not find/open datafile");
      
      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(fread(&cbuffer,1,1,datafile) != 1)
            {
              fclose(datafile);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      if (byteskip_ >= 0)
      {
        if (fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (fseek(datafile,sizes_[1]*sizes_[0]*elemsize_,SEEK_END)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
        
      if (fseek(datafile,elemsize_*sizes[0]_*coffset,SEEK_CUR)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      if (sizes[0]_*colread != fread((void *)buffer,elemsize_,sizes_[0]*colread,datafile))
      {
        fclose(datafile);
        throw TimeDataFileException("Error reading datafile");
      }
       
      fclose(datafile);
      }
    #else
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0) throw TimeDataFileException("Could not find/open datafile");

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              close(datafile_uni);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
    
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(sizes_[1])*static_cast<off_t>(sizes_[0])*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(sizes_[0])*static_cast<off_t>(coffset),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      size_t ret = ::read(datafile_uni,reinterpret_cast<void*>(buffer),static_cast<size_t>(elemsize_*sizes_[0]*colread));
      if (static_cast<size_t>(elemsize_*sizes[0]_*colread) != ret)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }
     
      ::close(datafile_uni);
    #endif
  
    buffer += (sizes_[0]*elemsize_*colread);
  }
  
  buffer = reinterpret_cast<char *>(mat->getData());
  
  if (swapbytes_) doswapbytes(reinterpret_cast<void*>(buffer),elemsize_,numcols*sizes_[0]);
  if (ntype_ == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(numcols*sizes_[0]-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]);}

  mh = dynamic_cast<Matrix *>(mat->transpose());
}

void getnrrd(NrrdDataHandle& nh,int dim, int dimstart, int dimend)
{

  FILE*                   datafile;
  int                     datafile_uni;

  if (dimstart > dimend) throw TimeDataFileException("Column start is bigger than column end");
  int numdims = dimend-dimstart + 1;

  SCIRun::NrrdData *nrrd = scinew SCIRun::NrrdData();
  if (nrrd == 0) throw TimeDataFileException("Could not allocate nrrd object");
  SCIRun::NrrdDataHandle handle = nrrd;

  nrrd->nrrd = nrrdNew();
  if (nrrd->nrrd == 0) throw TimeDataFileException("Could not allocate nrrd");

  size_t sizes[NRRD_DIM_MAX];
  for(int i=0; i<sizes_.size(); i++) sizes[i] = sizes_[i];
  nrrdAlloc_nva(nrrd->nrrd,ntype_,dimension_,sizes);
  if (nrrd->nrrd->data == 0) throw TimeDataFileException("Could not allocate nrrd");

  nrrdAxisInfoSet_nva(nrrd->nrrd,nrrdAxisInfoSpacing,dimension_,&(spacings_[0]));

  std::vector<double> t(dimension_);
  for (size_t p = 0; p < dimension_; p++) t[p] = 0.0;
  nrrdAxisInfoSet_nva(nrrd->nrrd,nrrdAxisInfoMin,dimension_,&(t[0]));  

  for (size_t p = 0; p < dimension_; p++) t[p] = if (spacings_[p] > 0) static_cast<double>(sizes_[p]*spacings_[p]));
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMax,dimension_,&(t[0]));

  std::vector<char *> tstr(dimension_);
  for (size_t p = 0; p < dimension_; p++) tstr[p] = kinds_.c_str();
  nrrdAxisInfoSet_nva(nrrd->nrrd,nrrdAxisInfoKind,dimension_,&(tstr[0]));  

  for (size_t p = 0; p < dimension_; p++) tstr[p] = units_.c_str();
  nrrdAxisInfoSet_nva(nrrd->nrrd,nrrdAxisInfoUnit,dimension_,&(tstr[0]));  

  char* buffer = reinterpret_cast<char *>(nrrd->nrrd->data);




}


void TimeDataFile::getcolnrrd(SCIRun::NrrdDataHandle& mh,int colstart,int colend)
{
  FILE*                   datafile;
  int                     datafile_uni;

  if (colstart > colend) throw TimeDataFileException("Column start is bigger than column end");

  int numcols = colend-colstart + 1;

  SCIRun::NrrdData *nrrd = scinew SCIRun::NrrdData();
  if (nrrd == 0) throw TimeDataFileException("Could not allocate nrrd object");
  SCIRun::NrrdDataHandle handle = nrrd;

  nrrd->nrrd = nrrdNew();
  if (nrrd->nrrd == 0) throw TimeDataFileException("Could not allocate nrrd");

  size_t size[NRRD_DIM_MAX];
  size[0] = nrows_;
  size[1] = numcols;
  nrrdAlloc_nva(nrrd->nrrd,ntype_,2,size);

  if (nrrd->nrrd->data == 0) throw TimeDataFileException("Could not allocate nrrd");

  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoSpacing,1.0,1.0);
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMin,0.0,0.0);  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMax,static_cast<double>(nrows_),static_cast<double>(numcols));
  
  char* buffer = reinterpret_cast<char *>(nrrd->nrrd->data);

  int c = colstart;
  while(c<=colend)
  {

    int coffset = c;
    int colread = (colend-c)+1;
    std::string fn = datafilename_;
    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (p=0;p<coloffset_.size();p++) { if((c >= coloffset_[p] )&&(c <coloffset_[p+1])) break;}
      if (p == coloffset_.size()) throw TimeDataFileException("Column index out of range");
      coffset = c-coloffset_[p];
      if (colend < coloffset_[p+1]) colread = (colend-c)+1; else colread = (coloffset_[p+1]-c);
      std::list<std::string>::iterator it = datafilenames_.begin();
      for (int q=0;q<p;q++) it++;
      fn = (*it);
    }
    
    c+=colread;
    
    #ifndef HAVE_UNISTD_H 
    
      datafile = fopen(fn.c_str(),"rb");
      if (datafile == 0) throw TimeDataFileException("Could not find/open datafile");
      
      if (lineskip_ > 0)
      {
         char buffer;
         int ln = lineskip_;
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
      
      if (byteskip_ >= 0)
      {
        if (fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (fseek(datafile,ncols_*nrows_*elemsize_,SEEK_END)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
        
      if (fseek(datafile,elemsize_*nrows_*coffset,SEEK_CUR)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      if (nrows_*colread != fread((void *)buffer,elemsize_,nrows_*colread,datafile))
      {
        fclose(datafile);
        throw TimeDataFileException("Error reading datafile");
      }
       
      fclose(datafile);
      }
    #else
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0) throw TimeDataFileException("Could not find/open datafile");

      if (lineskip_ > 0)
      {
         char buffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&buffer,1) != 1)
            {
              close(datafile_uni);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (buffer == '\n') ln--;
         }
      }
    
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(ncols_)*static_cast<off_t>(nrows_)*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(nrows_)*static_cast<off_t>(coffset),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      if (static_cast<size_t>(elemsize_*nrows_*colread) != ::read(datafile_uni,reinterpret_cast<void*>(buffer),static_cast<size_t>(elemsize_*nrows_*colread)))
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }
     
      ::close(datafile_uni);
    #endif
  
    buffer += (nrows_*elemsize_*colread);
  }
  
  if (swapbytes_) doswapbytes(reinterpret_cast<void*>(buffer),elemsize_,numcols*nrows_);

  mh = nrrd;
}


void TimeDataFile::getrowmatrix(SCIRun::MatrixHandle& mh,int rowstart,int rowend)
{
  FILE*                   datafile;
  int                     datafile_uni;

  if (rowstart > rowend) throw TimeDataFileException("Column start is bigger than column end");
  
  if (dimension_ > 2) throw TimeDataFileException("Matrix format does not support more than 2 dimensions");
		    
  int numrows =rowend-rowstart + 1;
  SCIRun::DenseMatrix *mat = scinew SCIRun::DenseMatrix(sizes_[1],numrows);
  SCIRun::MatrixHandle handle = dynamic_cast<SCIRun::Matrix *>(mat);
  
  if (mat == 0) throw TimeDataFileException("Could not allocate matrix");
  char* buffer = reinterpret_cast<char *>(mat->getData());  

  int c = 0;
  int cstart= 0;
  int cend = sizes_[1];
  std::string fn = datafilename_;
   
  while (c<sizes_[1])
  {

    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (;p<sdoffset_.size();p++) { if((c >= sdoffset_[p] )&&(c < sdoffset_[p+1])) break;}
      if (p == sdoffset_.size()) throw TimeDataFileException("Column index out of range");
      cstart = sdoffset_[p];
      cend = sdoffset_[p+1];
      c = sdoffset_[p+1];
      std::list<std::string>::iterator it = datafilenames_.begin();
      for (int q=0;q<p;q++) it++;
      fn = (*it);      
    }
    else
    {
      c = sizes_[1];
    }

    #ifndef HAVE_UNISTD_H
      datafile = fopen(fn.c_str(),"rb");
      if (datafile == 0) throw TimeDataFileException("Could not find/open datafile");

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(fread(&cbuffer,1,1,datafile) != 1)
            {
              fclose(datafile);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      

      if (byteskip_ >= 0)
      {
        if (fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (fseek(datafile,sizes_[1]*sizes_[0]*elemsize_,SEEK_END)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }

      if (fseek(datafile,elemsize_*rowstart,SEEK_CUR)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (numrows != fread(buffer+(elemsize_*j*numrows),elemsize_,numrows,datafile))
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (fseek(datafile,elemsize_*(sizes_[0]-numrows),SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      fclose(datafile);
    #else
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0) 
      {
        throw TimeDataFileException("Could not find/open datafile");
      }

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              ::close(datafile_uni);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      // Accoring to Gordon's definition
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(sizes_[1])*static_cast<off_t>(sizes_[0])*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(rowstart),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (static_cast<size_t>(numrows*elemsize_) != ::read(datafile_uni,buffer+(elemsize_*j*numrows),static_cast<size_t>(elemsize_*numrows)))
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(sizes_[0]-numrows),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      ::close(datafile_uni);  
    #endif
  }
    
  if (swapbytes_) doswapbytes(buffer,elemsize_,numrows*sizes[1]_);

  if (ntype_ == nrrdTypeChar) { char *fbuffer = reinterpret_cast<char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUChar) { unsigned char *fbuffer = reinterpret_cast<unsigned char *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeShort) { short *fbuffer = reinterpret_cast<short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUShort) { unsigned short *fbuffer = reinterpret_cast<unsigned short *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeInt) { int *fbuffer = reinterpret_cast<int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeUInt) { unsigned int *fbuffer = reinterpret_cast<unsigned int *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeFloat) { float *fbuffer = reinterpret_cast<float *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeLLong) { int64 *fbuffer = reinterpret_cast<int64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  if (ntype_ == nrrdTypeULLong) { uint64 *fbuffer = reinterpret_cast<uint64 *>(buffer); double *dbuffer = reinterpret_cast<double *>(buffer);  for (int j=(sizes_[1]*numrows-1);j>=0;j--) dbuffer[j] = static_cast<double>(fbuffer[j]); }
  
  mh = dynamic_cast<Matrix *>(mat->transpose());
}


void TimeDataFile::getrownrrd(SCIRun::NrrdDataHandle& mh,int rowstart,int rowend)
{
  FILE*                   datafile;
  int                     datafile_uni;

  if (rowstart > rowend) throw TimeDataFileException("Column start is bigger than column end");
    
  int numrows = rowend-rowstart + 1;
  
  SCIRun::NrrdData *nrrd = scinew SCIRun::NrrdData();
  if (nrrd == 0) throw TimeDataFileException("Could not allocate nrrd object");
  SCIRun::NrrdDataHandle handle = nrrd;

  nrrd->nrrd = nrrdNew();
  if (nrrd->nrrd == 0)  throw TimeDataFileException("Could not allocate nrrd");

  size_t size[NRRD_DIM_MAX];
  size[0] = numrows;
  size[1] = ncols_;
  nrrdAlloc_nva(nrrd->nrrd,ntype_,2,size);
  if (nrrd->nrrd->data == 0) throw TimeDataFileException("Could not allocate nrrd");
  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoSpacing,1.0,1.0);
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMin,0.0,0.0);  
  nrrdAxisInfoSet(nrrd->nrrd,nrrdAxisInfoMax,static_cast<double>(numrows),static_cast<double>(ncols_));
  
  char* buffer = reinterpret_cast<char *>(nrrd->nrrd->data);  
    
  int c =0;
  while (c<ncols_)
  {
    std::string fn = datafilename_;
    int cstart= 0;
    int cend = ncols_;
    if (datafilenames_.size() > 0)
    {
      // find file to read
      int  p=0;
      for (p=0;p<coloffset_.size();p++) { if((c >= coloffset_[p] )&&(c <coloffset_[p+1])) break;}
      if (p == coloffset_.size()) throw TimeDataFileException("Column index out of range");
      cstart = coloffset_[p];
      cend = coloffset_[p+1];
      c = coloffset_[p+1];
      std::list<std::string>::iterator it = datafilenames_.begin();
      for (int q=0;q<p;q++) it++;
      fn = (*it);      
    }
    else
    {
      c = ncols_;
    }

    #ifndef HAVE_UNISTD_H
      datafile = fopen(fn.c_str(),"rb");
      if (datafile == 0) throw TimeDataFileException("Could not find/open datafile");

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(fread(&cbuffer,1,1,datafile) != 1)
            {
              fclose(datafile);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      

      if (byteskip_ >= 0)
      {
        if (fseek(datafile,byteskip_,SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (fseek(datafile,ncols_*nrows_*elemsize_,SEEK_END)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Could not read datafile");
        }
      }

    
      if (fseek(datafile,elemsize_*rowstart,SEEK_CUR)!=0)
      {
        fclose(datafile);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

      int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (numrows != fread(buffer+(elemsize_*j*numrows),elemsize_,numrows,datafile))
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (fseek(datafile,elemsize_*(nrows_-numrows),SEEK_CUR)!=0)
        {
          fclose(datafile);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      fclose(datafile);
    #else
      datafile_uni = ::open(fn.c_str(),O_RDONLY|O_LARGEFILE,0);
      if (datafile_uni < 0) 
      {
        throw TimeDataFileException("Could not find/open datafile");
      }

      if (lineskip_ > 0)
      {
         char cbuffer;
         int ln = lineskip_;
         while (ln)
         {
            if(::read(datafile_uni,&cbuffer,1) != 1)
            {
              ::close(datafile_uni);
              throw TimeDataFileException("Could not read header of datafile"); 
            }
            if (cbuffer == '\n') ln--;
         }
      }
      
      // Accoring to Gordon's definition
      if (byteskip_ >= 0)
      {
        if (::lseek(datafile_uni,static_cast<off_t>(byteskip_),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
      else
      {
        if (::lseek(datafile_uni,static_cast<off_t>(ncols_)*static_cast<off_t>(nrows_)*static_cast<off_t>(elemsize_),SEEK_END)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Could not read datafile");
        }
      }
    
      if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(rowstart),SEEK_CUR)<0)
      {
        ::close(datafile_uni);
        throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
      }    

       int i,j;
      for (i=0, j=cstart;j<cend;i++,j++)
      {
        if (static_cast<size_t>(numrows*elemsize_) != ::read(datafile_uni,buffer+(elemsize_*j*numrows),static_cast<size_t>(elemsize_*numrows)))
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }
        if (::lseek(datafile_uni,static_cast<off_t>(elemsize_)*static_cast<off_t>(nrows_-numrows),SEEK_CUR)<0)
        {
          ::close(datafile_uni);
          throw TimeDataFileException("Improper data file, check number of columns and rows in header file");
        }    
      }
      ::close(datafile_uni);  
    #endif
  }
    
  if (swapbytes_) doswapbytes(buffer,elemsize_,numrows*ncols_);

  mh = nrrd;
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
  bool quote_on =false;
  
  for (std::string::size_type i=0;i<str.size();i++) 
  {
    if (str[i]=='"')
    {
      if (quote_on == false) quote_on = true; else quote_on = false;
    }
    if (quote_on) newstr += str[i]; else if ((str[i] != ' ' )&&(str[i] != '\t')&&(str[i] != '\n')&&(str[i] != '\r')) newstr += str[i];
  }
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

