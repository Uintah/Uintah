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
 * FILE: matfile.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 9 JAN 2004
 */
 
/*
 * The matfile class is the basic interface for reading and
 * writing .mat files (matlab files). This class currently
 * only supports the so called matlab version 5 file format
 * which is used from version 5.0 to version 6.5, currently 
 * the lastest version of matlab available. Matlab V4 files
 * should be converted using matlab into the newer file format.
 *
 * This class handles the following aspects:
 * - opening and closing .mat files
 * - handling byteswapping
 * - reading/writing the file header
 * - reading/writing the tags in the .mat file
 *
 */
 
#include "matfile.h"
#include "matfiledata.h"

 
namespace MatlabIO {

// Function for doing byteswapping when loading a file created on a different platform 
  
void matfile::mfswapbytes(void *vbuffer,long elsize,long size)
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
        throw unknown_type();
     	break; 
   }  
}

// my read and write functions that check for errors and create exceptions
// these functions invoke byteswapping and decrease the amount of coding in
// the more dedicated read and write functions.

void matfile::mfwrite(void *buffer,long elsize,long size)
{
	FILE *fptr;
	fptr = m_->fptr_;
	
    if (fptr == 0) return;
    if (static_cast<long>(fwrite(buffer,elsize,size,fptr)) != size) throw io_error();
    if (ferror(fptr)) throw io_error();
}

void matfile::mfwrite(void *buffer,long elsize,long size,long offset)
{
    FILE *fptr;
	fptr = m_->fptr_;
	
	if (fptr == 0) return;
    if (fseek(fptr,offset,SEEK_SET) != 0) throw io_error();
    if (ferror(fptr)) throw io_error();
    if (static_cast<long>(fwrite(buffer,elsize,size,fptr)) != size) throw io_error();
    if (ferror(fptr)) throw io_error();
}

void matfile::mfread(void *buffer,long elsize,long size)
{
	FILE *fptr;
	fptr = m_->fptr_;
	
    if (fptr == 0) return;
    if (static_cast<long>(fread(buffer,elsize,size,fptr)) != size) throw io_error();
    if (ferror(fptr)) throw io_error();
	if (m_->byteswap_) mfswapbytes(buffer,elsize,size);
}

void matfile::mfread(void *buffer,long elsize,long size,long offset)
{
	FILE *fptr;
	fptr = m_->fptr_;
	
    if (fptr == 0) return;
    if (fseek(fptr,offset,SEEK_SET) != 0) throw io_error();
    if (ferror(fptr)) throw io_error();
    if (static_cast<long>(fread(buffer,elsize,size,fptr)) != size) throw io_error();
    if (ferror(fptr)) throw io_error();
    if (m_->byteswap_) mfswapbytes(buffer,elsize,size);
}


// Separate functions for reading and writing the header

void matfile::mfwriteheader()
{
	short endian = 19785;	// In characters this is 'MI'
    
    if (m_->fptr_ == 0) return;
    mfwrite(static_cast<void *>(&(m_->headertext_[0])),sizeof(char),124,0);    
    mfwrite(static_cast<void *>(&(m_->version_)),sizeof(short),1,124);
    mfwrite(static_cast<void *>(&endian),sizeof(short),1,126);
}

void matfile::mfreadheader()
{
    short endian; 
   
    // Determine the endian of the file. Byte 127/128 should contain the letters 'MI', and 'IM' if
    // the bytes in the file are swapped. If neihter of this combination is found the file must
    // be of a different type
    
    m_->byteswap_ = 0;
    mfread(static_cast<void *>(&endian),1,sizeof(short),126);
    if (endian != 19785) if (endian == 18765) m_->byteswap_ = 1; else throw invalid_file_format();

    mfread(static_cast<void *>(&(m_->headertext_[0])),sizeof(char),124,0);
    mfread(static_cast<void *>(&(m_->version_)),sizeof(short),1,124);
}

// PUBLIC FUNCTIONS

// constructors
matfile::matfile()
    : m_(0)
{
	m_ = new mxfile;
	m_->fptr_ = 0;
	m_->byteswap_ = 0;
	m_->ref_ = 1;
}
 
matfile::matfile(std::string filename,std::string mode)
    : m_(0)
{
	m_ = new mxfile;
	m_->fptr_ = 0;
	m_->byteswap_ = 0;
	m_->ref_ = 1;
	
    open(filename,mode);
} 

//destructor

matfile::~matfile()
{
	m_->ref_--;
	if (m_->ref_ == 0)
	{
		if (m_->fptr_) close();
		delete m_;
	}
	m_ = 0;
}

// copy and assignment operators

matfile::matfile(const matfile &mf)
{
	m_ = mf.m_;
    m_->ref_++;
}
        
matfile& matfile::operator= (const matfile &mf)
{
    if (this != &mf)
    {
		if (m_ != 0)
		{
			m_->ref_--;
			if (m_->ref_ == 0)
			{
				if (m_->fptr_) close();
				delete m_;
			}
			m_ = 0;
		}
        m_ = mf.m_;
        m_->ref_++;
    }
    return *this;
}

// header read and write interface

std::string matfile::getheadertext()
{
    std::string text;
    m_->headertext_[124] = 0;		// make sure it is zero terminated
    text = m_->headertext_;			// should convert it to string class
    return(text);
}

void matfile::setheadertext(std::string text)
{
    long len;
    const char *ctext = text.c_str();
	
	len = static_cast<long>(text.size());
    if (len > 124) len = 124;		// Protection against textheaders that are too long
    
	for (long p=0;p<len;p++) m_->headertext_[p] = ctext[p];
}



void matfile::open(std::string filename,std::string mode)
{
    try
    {
        m_->fname_ = filename;
        m_->fmode_ = mode;
        if (isreadaccess())
        {
            char fileid[4];
            
            if (!(m_->fptr_ = fopen(m_->fname_.c_str(),"rb"))) throw could_not_open_file();
            
            // Determine file length, file needs to contain at least the 128 byte header
            if (fseek(m_->fptr_,0,SEEK_END) != 0) throw io_error();
            m_->flength_ = ftell(m_->fptr_);
            if (m_->flength_ < 128) throw invalid_file_format(); 
    
            // Determine whether file is of a different type
            // There are many .mat files out there
            // .mat Matlab version 4 files start with 4 zero bytes
            // .mat SCIRun files start with 'SCI\n'
            // .mat Matlab version 5 and higher files start with text, but probably not 'SCI\n'
            //      the latter files are identified by the 'MI' at byte 127 and 128 
            
            mfread(static_cast<void *>(&fileid[0]),4,sizeof(char),0);
            if ((fileid[0] == 0)&&(fileid[1] == 0)&&(fileid[2] == 0)&&(fileid[3] == 0)) throw invalid_file_format(); 
            if ((fileid[0] == 'S')&&(fileid[1] == 'C')&&(fileid[2] == 'I')&&(fileid[3] == '\n')) throw invalid_file_format(); 
    
            mfreadheader();
            m_->curptr_.hdrptr = 128;
            m_->curptr_.datptr = -1;
            m_->curptr_.startptr = 128;
            m_->curptr_.endptr = m_->flength_;
            m_->curptr_.size   = 0;
        }
        else if (iswriteaccess())
        {
            if(!(m_->fptr_ = fopen(m_->fname_.c_str(),"wb"))) throw could_not_open_file();
            
            // Setup the default parameters for the header
            // Version 0x0100 is the matlab default for an externaly created std::FILE
            // All files are written in the native format, byteswapping is done
            // when reading files.
            
            m_->version_ = 0x0100;
            m_->byteswap_ = 0;
                	
            for(long p=0;p<124;p++) m_->headertext_[p] = 0;
            strcpy(&(m_->headertext_[0]),"MATLAB 5.0 MAT-FILE");
            
            mfwriteheader();
            m_->curptr_.hdrptr = 128;
            m_->curptr_.datptr = -1;
            m_->curptr_.startptr = 128;
            m_->curptr_.endptr = -1;
            m_->curptr_.size   = 0;
        }
        else
        {
            throw could_not_open_file();
        }
    }
    catch (...)
    {
        if (m_->fptr_) { fclose(m_->fptr_); m_->fptr_ = 0; }
        throw;
    }
}

void matfile::close()
{
    if (m_->fptr_ == 0) return;	// file is already closed
    try
    {
        while (m_->ptrstack_.size()) { m_->ptrstack_.pop();}	// empty pointer stack
        if (iswriteaccess()) mfwriteheader();
        fclose(m_->fptr_); m_->fptr_ = 0;
    }
    catch (...)
    {	// if writeheader failed, force the std::FILE to close 
        if (m_->fptr_) { fclose(m_->fptr_); m_->fptr_ = 0; }
        throw;
    }
}


// rewind the complete file
void matfile::rewind()
{
    while (m_->ptrstack_.size()) { m_->ptrstack_.pop();}
    if (iswriteaccess())
    {
            m_->curptr_.hdrptr = 128;
            m_->curptr_.datptr = -1;
            m_->curptr_.endptr = -1;
            m_->curptr_.size   = 0;
    }
    
	if (isreadaccess())
    {
            m_->curptr_.hdrptr = 128;
            m_->curptr_.datptr = -1;
            m_->curptr_.endptr = m_->flength_;
            m_->curptr_.size   = 0;
    }
}

long matfile::nexttag()
{
    if (m_->curptr_.hdrptr == m_->curptr_.endptr) return(0);
    if (m_->curptr_.datptr == -1) 
	{
		if (isreadaccess())
		{   // try to read next one
			matfiledata md;
			readtag(md);
			if (m_->curptr_.datptr == -1) 
			{   // reading next one failed
				// return current one
				return(m_->curptr_.hdrptr);
			}	
	
		}
		if (iswriteaccess())
		{
			return(m_->curptr_.hdrptr);
		}
    }
	
	m_->curptr_.hdrptr = m_->curptr_.datptr+m_->curptr_.size;
    m_->curptr_.hdrptr = (((m_->curptr_.hdrptr-1)/8)+1)*8;	// hdrptr cannot be zero or negative, so this should give the proper alignment
    m_->curptr_.datptr = -1;
    m_->curptr_.size = 0;
    if ( isreadaccess()) { if (m_->curptr_.hdrptr > m_->curptr_.endptr) m_->curptr_.hdrptr = m_->curptr_.endptr; }

    if (m_->curptr_.hdrptr ==  m_->curptr_.endptr) { return(0); } else { return( m_->curptr_.hdrptr); }
}

bool matfile::openchild()
{
    matfileptr childptr;
    if (m_->curptr_.datptr == -1) return(false);
    childptr.hdrptr = m_->curptr_.datptr;
    childptr.startptr = m_->curptr_.datptr;
    childptr.endptr = m_->curptr_.datptr+m_->curptr_.size;
    childptr.datptr = -1;
    childptr.size   = 0;
 
	if (iswriteaccess()) { childptr.endptr = -1;}
	   
    m_->ptrstack_.push(m_->curptr_);
    m_->curptr_ = childptr;
    return(true);
}

void matfile::closechild()
{
    matfileptr parptr;
    
    if (m_->ptrstack_.empty()) return;		// We are already at the top level;
    parptr = m_->ptrstack_.top();
	
    if (iswriteaccess())
    {
        long segsize;
        if (m_->curptr_.datptr != -1) nexttag();
		
        segsize = m_->curptr_.hdrptr-parptr.datptr;
		
		m_->ptrstack_.pop();
        m_->curptr_ = parptr;    
        mfwrite(static_cast<void *>(&segsize),sizeof(long),1,m_->curptr_.hdrptr+4);
		m_->curptr_.size = segsize;
    }
    else
    {
        m_->ptrstack_.pop();
        m_->curptr_ = parptr; 
    }
}

long matfile::firsttag()
{
    m_->curptr_.hdrptr = m_->curptr_.startptr;
    m_->curptr_.datptr = -1;
    m_->curptr_.size = 0;
    if (m_->curptr_.hdrptr == m_->curptr_.endptr) return(0); else return(m_->curptr_.hdrptr);
}

long matfile::gototag(long tagaddress)
{
    m_->curptr_.hdrptr = tagaddress;
    m_->curptr_.datptr = -1;
    m_->curptr_.size  = 0;
    if (m_->curptr_.hdrptr == m_->curptr_.endptr) return(0); else return(m_->curptr_.hdrptr);
}


void matfile::readtag(matfiledata& md)
{
    long  size = 0;
    long  type = 0;
    
    md.clear();
    try
    {
        if (m_->curptr_.hdrptr == m_->curptr_.endptr) return;
        
        mfread(static_cast<void *>(&type),sizeof(long),1,m_->curptr_.hdrptr);
        mfread(static_cast<void *>(&size),sizeof(long),1,m_->curptr_.hdrptr+4);
        m_->curptr_.datptr = m_->curptr_.hdrptr+8;
 
        if (type > miMATRIX)
        {   // Hence it must be a compressed data tag
            short	csizetype[2];
            // trick the reader in reading a 32bit long, so both 16bit integers
            // are in the proper order.
            mfread(static_cast<void *>(&(csizetype[0])),sizeof(long),1,m_->curptr_.hdrptr);
            if (byteswapmachine())
			{
				size = static_cast<long>(csizetype[1]);
				type = static_cast<long>(csizetype[0]);			
			}
			else
			{
				size = static_cast<long>(csizetype[0]);
				type = static_cast<long>(csizetype[1]);
            }
            m_->curptr_.datptr = m_->curptr_.hdrptr+4;
        }
        m_->curptr_.size = size;


        // If type still invalid then something else is going on
        // Throw an exception as we cannot read this field
        if (type > miMATRIX) throw unknown_type();
        
        md.clear();
        md.type(static_cast<mitype>(type));
        
    }
    catch(...)
    {
        md.clear();		// Clean up everything allocated in this function
        throw;
    }
}

void matfile::readdat(matfiledata& md)
{
    long  size = 0;
    long  type = 0;
    
    md.clear();
    try
    {
        if (m_->curptr_.hdrptr == m_->curptr_.endptr) return;
        
        mfread(static_cast<void *>(&type),sizeof(long),1,m_->curptr_.hdrptr);
        mfread(static_cast<void *>(&size),sizeof(long),1,m_->curptr_.hdrptr+4);
        m_->curptr_.datptr = m_->curptr_.hdrptr+8;
 
        if (type > miMATRIX)
        {   // Hence it must be a compressed data tag
            short	csizetype[2];
            // trick the reader in reading a 32bit long, so both 16bit integers
            // are in the proper order.
            mfread(static_cast<void *>(&(csizetype[0])),sizeof(long),1,m_->curptr_.hdrptr);
            if (byteswapmachine())
			{
				size = static_cast<long>(csizetype[1]);
				type = static_cast<long>(csizetype[0]);			
			}
			else
			{
				size = static_cast<long>(csizetype[0]);
				type = static_cast<long>(csizetype[1]);
            }
			m_->curptr_.datptr = m_->curptr_.hdrptr+4;
        }
        m_->curptr_.size = size;

        // If type still invalid then something else is going on
        // Throw an exception as we cannot read this field
        if (type > miMATRIX) throw unknown_type();
        
        md.newdatabuffer(size,static_cast<mitype>(type));
        if (md.size() > 0) mfread(md.databuffer(),md.elsize(),md.size(),m_->curptr_.datptr);
		
    }
    catch(...)
    {
        md.clear();		// Clean up the memory allocated for the data
        throw;
    }
	
	
}

void matfile::writetag(matfiledata& md)
{
    // Since the miMATRIX class is compiled out of different tags
    // that are written after this tag, the size field needs to be
    // updated afterwards. Since the size is unknown we cannot assume
    // it is smaller 256. Hence we must exclude the miMATRIX class
    
    if ((md.type() != miMATRIX)&&(md.bytesize() < 5))
    {	// write a compressed header
        short	csizetype[2];
        csizetype[0] = static_cast<short>(md.bytesize());
        csizetype[1] = static_cast<short>(md.type());
        mfwrite(static_cast<void *>(&(csizetype[0])),sizeof(long),1,m_->curptr_.hdrptr);
        m_->curptr_.datptr = m_->curptr_.hdrptr+4;
        m_->curptr_.size = md.bytesize();
    }
    else
    {   // write a normal header     
        long size = md.bytesize();
        long type = static_cast<long>(md.type());
        mfwrite(static_cast<void *>(&type),sizeof(long),1,m_->curptr_.hdrptr);
        mfwrite(static_cast<void *>(&size),sizeof(long),1,m_->curptr_.hdrptr+4);
        m_->curptr_.datptr = m_->curptr_.hdrptr+8;
        m_->curptr_.size = size;
    }  
}

void matfile::writedat(matfiledata& md)
{
    writetag(md);
    if (md.size() > 0) mfwrite(md.databuffer(),md.elsize(),md.size(),m_->curptr_.datptr);
	
	// fill the datablock with zeros until it is alligned with 8 byte boundary
	long remsize = 0;
	if (md.bytesize() > 0) remsize = ((((md.bytesize()-1)/8)+1)*8)-md.bytesize();
	
	if (remsize)
	{   // fill up remaining bytes
		long buffer[2] = {0,0};
		mfwrite(static_cast<void *>(&buffer[0]),1,remsize,m_->curptr_.datptr+md.bytesize());
	}
}

bool matfile::isreadaccess()
{
    if (m_->fmode_.compare("r") == 0) return(true);
    return(false);	
}

bool matfile::iswriteaccess()
{
    if (m_->fmode_.compare("w") == 0) return(true);
    return(false);	
}

bool matfile::byteswap()
{
	if (m_->byteswap_ == 1) return(true);
	return(false);
}

bool matfile::byteswapmachine()
{
	short test = 0x00FF;
	char *ptr;
	// make the compiler happy
	ptr = static_cast<char *>(static_cast<void *>(&test));
	if (ptr[0]) return(true);
	return(false);
}

} // end namespace
