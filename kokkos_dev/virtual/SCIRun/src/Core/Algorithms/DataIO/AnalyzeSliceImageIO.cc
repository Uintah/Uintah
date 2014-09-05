/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  AnalyzeSliceImageIO : Inherit from itkAnalyzeSliceImageIO
 *               to allow for reading a slice at a time.
 *
 *  Written by:
 *   Darby Van Uitert
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2005
 *
 */


#include <Core/Algorithms/DataIO/AnalyzeSliceImageIO.h>
#include "itkIOCommon.h"
#include "itkExceptionObject.h"
#include "itkByteSwapper.h"
#include "itkMetaDataObject.h"

#include <itksys/SystemTools.hxx>

#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>

namespace itk
{

const char *const ANALYZESLICE_ScanNumber = "ANALYZE_ScanNumber";
const char *const ANALYZESLICE_O_MAX = "ANALYZE_O_MAX";
const char *const ANALYZESLICE_O_MIN = "ANALYZE_O_MIN";
const char *const ANALYZESLICE_S_MAX = "ANALYZE_S_MAX";
const char *const ANALYZESLICE_S_MIN = "ANALYZE_S_MIN";
const char *const ANALYZESLICE_CAL_MAX = "ANALYZE_CAL_MAX";
const char *const ANALYZESLICE_CAL_MIN = "ANALYZE_CAL_MIN";
const char *const ANALYZESLICE_GLMAX = "ANALYZE_GLMAX";
const char *const ANALYZESLICE_GLMIN = "ANALYZE_GLMIN";
const char *const ANALYZESLICE_AUX_FILE_NAME = "ANALYZE_AUX_FILE_NAME";
const char *const ANALYZESLICE_CALIBRATIONUNITS = "ANALYZE_CALIBRATIONUNITS";
//An array of the Analyze v7.5 known DataTypes
const char SCIDataTypes[12][10]=  {
  "UNKNOWN","BINARY","CHAR","SHORT", "INT","FLOAT",
  "COMPLEX", "DOUBLE","RGB","ALL","USHORT","UINT"
};

//An array with the corresponding number of bits for each image type.
//NOTE: the following two line should be equivalent.
const short int SCIDataTypeSizes[12]={0,1,8,16,32,32,64,64,24,0,16,32};

//An array with Data type key sizes
const short int SCIDataTypeKey[12]={
  ANALYZE_DT_UNKNOWN,
  ANALYZE_DT_BINARY,
  ANALYZE_DT_UNSIGNED_CHAR,
  ANALYZE_DT_SIGNED_SHORT,
  ANALYZE_DT_SIGNED_INT,
  ANALYZE_DT_FLOAT,
  ANALYZE_DT_COMPLEX,
  ANALYZE_DT_DOUBLE,
  ANALYZE_DT_RGB,
  ANALYZE_DT_ALL,
  SPMANALYZE_DT_UNSIGNED_SHORT,
  SPMANALYZE_DT_UNSIGNED_INT
};


//GetExtension from uiig library.
static std::string
GetExtension( const std::string& filename ) {

  // This assumes that the final '.' in a file name is the delimiter
  // for the file's extension type
  const std::string::size_type it = filename.find_last_of( "." );

  // This determines the file's type by creating a new string
  // who's value is the extension of the input filename
  // eg. "myimage.gif" has an extension of "gif"
  std::string fileExt( filename, it+1, filename.length() );

  return( fileExt );
}


//GetRootName from uiig library.
static std::string
GetRootName( const std::string& filename )
{
  const std::string fileExt = GetExtension(filename);

  // Create a base filename
  // i.e Image.hdr --> Image
  if( fileExt.length() > 0 )
    {
    const std::string::size_type it = filename.find_last_of( fileExt );
    std::string baseName( filename, 0, it-fileExt.length() );
    return( baseName );
    }
  else
    {
    // Case when the extension is nothing (Analyze)
    return( filename );
    }
}


static std::string
GetHeaderFileName( const std::string & filename )
{
  std::string ImageFileName = GetRootName(filename);
  std::string fileExt = GetExtension(filename);
  //If file was named xxx.img.gz then remove both the gz and the img endings.
  if(!fileExt.compare("gz"))
    {
    ImageFileName=GetRootName(GetRootName(filename));
    }
  ImageFileName += ".hdr";
  return( ImageFileName );
}

//Returns the base image filename.
static std::string GetImageFileName( const std::string& filename )
{
  // Why do we add ".img" here?  Look in fileutils.h
  std::string fileExt = GetExtension(filename);
  std::string ImageFileName = GetRootName(filename);
  if(!fileExt.compare("gz"))
    {
    //First strip both extensions off
    ImageFileName=GetRootName(GetRootName(filename));
    ImageFileName += ".img.gz";
    }
  else if(!fileExt.compare("img") || !fileExt.compare("hdr") )
    {
    ImageFileName += ".img";
    }
  else
    {
    //uiig::Reporter* reporter = uiig::Reporter::getReporter();
    //std::string temp="Error, Can not determine compressed file image name. ";
    //temp+=filename;
    //reporter->setMessage( temp );
    return ("");
    }
  return( ImageFileName );
}


void
AnalyzeSliceImageIO::SwapBytesIfNecessary( void* buffer,
                                      unsigned long numberOfPixels )
{
  if ( m_ByteOrder == LittleEndian )
    {
    switch(m_ComponentType)
      {
      case CHAR:
        ByteSwapper<char>::SwapRangeFromSystemToLittleEndian((char*)buffer,
                                                             numberOfPixels );
        break;
      case UCHAR:
        ByteSwapper<unsigned char>::SwapRangeFromSystemToLittleEndian
          ((unsigned char*)buffer, numberOfPixels );
        break;
      case SHORT:
        ByteSwapper<short>::SwapRangeFromSystemToLittleEndian
          ((short*)buffer, numberOfPixels );
        break;
      case USHORT:
        ByteSwapper<unsigned short>::SwapRangeFromSystemToLittleEndian
          ((unsigned short*)buffer, numberOfPixels );
        break;
      case INT:
        ByteSwapper<int>::SwapRangeFromSystemToLittleEndian
          ((int*)buffer, numberOfPixels );
        break;
      case UINT:
        ByteSwapper<unsigned int>::SwapRangeFromSystemToLittleEndian
          ((unsigned int*)buffer, numberOfPixels );
        break;
      case LONG:
        ByteSwapper<long>::SwapRangeFromSystemToLittleEndian
          ((long*)buffer, numberOfPixels );
        break;
      case ULONG:
        ByteSwapper<unsigned long>::SwapRangeFromSystemToLittleEndian
          ((unsigned long*)buffer, numberOfPixels );
        break;
      case FLOAT:
        ByteSwapper<float>::SwapRangeFromSystemToLittleEndian((float*)buffer,
                                                              numberOfPixels );
        break;
      case DOUBLE:
        ByteSwapper<double>::SwapRangeFromSystemToLittleEndian
          ((double*)buffer, numberOfPixels );
        break;
      default:
        ExceptionObject exception(__FILE__, __LINE__);
        exception.SetDescription("Pixel Type Unknown");
        throw exception;
      }
    }
  else
    {
    switch(m_ComponentType)
      {
      case CHAR:
        ByteSwapper<char>::SwapRangeFromSystemToBigEndian((char *)buffer,
                                                          numberOfPixels );
        break;
      case UCHAR:
        ByteSwapper<unsigned char>::SwapRangeFromSystemToBigEndian
          ((unsigned char *)buffer, numberOfPixels );
        break;
      case SHORT:
        ByteSwapper<short>::SwapRangeFromSystemToBigEndian
          ((short *)buffer, numberOfPixels );
        break;
      case USHORT:
        ByteSwapper<unsigned short>::SwapRangeFromSystemToBigEndian
          ((unsigned short *)buffer, numberOfPixels );
        break;
      case INT:
        ByteSwapper<int>::SwapRangeFromSystemToBigEndian
          ((int *)buffer, numberOfPixels );
        break;
      case UINT:
        ByteSwapper<unsigned int>::SwapRangeFromSystemToBigEndian
          ((unsigned int *)buffer, numberOfPixels );
        break;
      case LONG:
        ByteSwapper<long>::SwapRangeFromSystemToBigEndian
          ((long *)buffer, numberOfPixels );
        break;
      case ULONG:
        ByteSwapper<unsigned long>::SwapRangeFromSystemToBigEndian
          ((unsigned long *)buffer, numberOfPixels );
        break;
      case FLOAT:
        ByteSwapper<float>::SwapRangeFromSystemToBigEndian
          ((float *)buffer, numberOfPixels );
        break;
      case DOUBLE:
        ByteSwapper<double>::SwapRangeFromSystemToBigEndian
          ((double *)buffer, numberOfPixels );
        break;
      default:
        ExceptionObject exception(__FILE__, __LINE__);
        exception.SetDescription("Pixel Type Unknown");
        throw exception;
      }
    }
}

ImageIOBase::ByteOrder
AnalyzeSliceImageIO::CheckAnalyzeEndian(const struct dsr &temphdr)
{
  ImageIOBase::ByteOrder returnvalue;
  // Machine and header endianess is same

  //checking hk.extents only is NOT a good idea. Many programs do not set
  //hk.extents correctly. Doing an additional check on hk.sizeof_hdr
  //increases chance of correct result. --Juerg Tschirrin Univeristy of Iowa
  //All properly constructed analyze images should have the extents feild
  //set.  It is part of the file format standard.  While most headers of
  //analyze images are 348 bytes long, The Analyze file format allows the
  //header to have other lengths.
  //This code will fail in the unlikely event that the extents feild is
  //not set (invalid anlyze file anyway) and the header is not the normal
  //size.  Other peices of code have used a heuristic on the image
  //dimensions.  If the Image dimensions is greater
  //than 16000 then the image is almost certainly byte-swapped-- Hans

  const ImageIOBase::ByteOrder systemOrder= (ByteSwapper<int>::SystemIsBigEndian()) ? BigEndian : LittleEndian;

  if((temphdr.hk.extents == 16384) || (temphdr.hk.sizeof_hdr == 348))
    {
    returnvalue = systemOrder;
    }
  else
    {
    // File does not match machine
    returnvalue = (systemOrder == BigEndian ) ? LittleEndian : BigEndian;
    }
  return returnvalue;
}

void
AnalyzeSliceImageIO::SwapHeaderBytesIfNecessary( struct dsr * const imageheader )
{
  if ( m_ByteOrder == LittleEndian )
    {
    // NOTE: If machine order is little endian, and the data needs to be
    // swapped, the SwapFromBigEndianToSystem is equivalent to
    // SwapFromSystemToBigEndian.
    ByteSwapper<int>::SwapFromSystemToLittleEndian(&imageheader->hk.sizeof_hdr);
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( (&imageheader->hk.data_type[0]),10 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.db_name[0]),18 );
    ByteSwapper<int  >::SwapFromSystemToLittleEndian( &imageheader->hk.extents );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian( &imageheader->hk.session_error );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToLittleEndian( &imageheader->hk.regular );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToLittleEndian( &imageheader->hk.hkey_un0 );

    ByteSwapper<short int>::SwapRangeFromSystemToLittleEndian( &imageheader->dime.dim[0], 8 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.vox_units[0]),4 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.cal_units[0]),8 );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian( &imageheader->dime.unused1 );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian( &imageheader->dime.datatype );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian( &imageheader->dime.bitpix );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian( &imageheader->dime.dim_un0 );

    ByteSwapper<float>::SwapRangeFromSystemToLittleEndian( &imageheader->dime.pixdim[0],8 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.vox_offset );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.roi_scale );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.funused1 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.funused2 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.cal_max );
    ByteSwapper<float>::SwapFromSystemToLittleEndian( &imageheader->dime.cal_min );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->dime.compressed );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->dime.verified );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->dime.glmax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->dime.glmin );

    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.descrip[0]),80 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.aux_file[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToLittleEndian( &(imageheader->hk.orient );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.originator[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.generated[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.scannum[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.patient_id[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.exp_date[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.exp_time[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToLittleEndian( &(imageheader->hk.hist_un0[0]),24 );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.views );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.vols_added );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.start_field );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.field_skip );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.omax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.omin );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.smax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian( &imageheader->hist.smin );
    }
  else if ( m_ByteOrder == BigEndian )
    {
    //NOTE: If machine order is little endian, and the data needs to be
    // swapped, the SwapFromBigEndianToSystem is equivalent to
    // SwapFromSystemToLittleEndian.
    ByteSwapper<int  >::SwapFromSystemToBigEndian( &imageheader->hk.sizeof_hdr );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( (&imageheader->hk.data_type[0]),10 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.db_name[0]),18 );
    ByteSwapper<int  >::SwapFromSystemToBigEndian( &imageheader->hk.extents );
    ByteSwapper<short int>::SwapFromSystemToBigEndian( &imageheader->hk.session_error );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToBigEndian( &imageheader->hk.regular );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToBigEndian( &imageheader->hk.hkey_un0 );

    ByteSwapper<short int>::SwapRangeFromSystemToBigEndian( &imageheader->dime.dim[0], 8 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.vox_units[0]),4 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.cal_units[0]),8 );
    ByteSwapper<short int>::SwapFromSystemToBigEndian( &imageheader->dime.unused1 );
    ByteSwapper<short int>::SwapFromSystemToBigEndian( &imageheader->dime.datatype );
    ByteSwapper<short int>::SwapFromSystemToBigEndian( &imageheader->dime.bitpix );
    ByteSwapper<short int>::SwapFromSystemToBigEndian( &imageheader->dime.dim_un0 );

    ByteSwapper<float>::SwapRangeFromSystemToBigEndian( &imageheader->dime.pixdim[0],8 );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.vox_offset );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.roi_scale );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.funused1 );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.funused2 );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.cal_max );
    ByteSwapper<float>::SwapFromSystemToBigEndian( &imageheader->dime.cal_min );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->dime.compressed );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->dime.verified );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->dime.glmax );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->dime.glmin );

    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.descrip[0]),80 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.aux_file[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapFromSystemToBigEndian( &(imageheader->hk.orient );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.originator[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.generated[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.scannum[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.patient_id[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.exp_date[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.exp_time[0]),24 );
    //Here for completeness ByteSwapper<char >::SwapRangeFromSystemToBigEndian( &(imageheader->hk.hist_un0[0]),24 );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.views );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.vols_added );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.start_field );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.field_skip );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.omax );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.omin );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.smax );
    ByteSwapper<int>::SwapFromSystemToBigEndian( &imageheader->hist.smin );
    }
  else
    {
    ExceptionObject exception(__FILE__, __LINE__);
    exception.SetDescription("Machine Endian Type Unknown");
    throw exception;
    }
}


AnalyzeSliceImageIO::AnalyzeSliceImageIO()
{
  //by default, only have 3 dimensions
  this->SetNumberOfDimensions(3);
  m_PixelType         = SCALAR;
  m_ComponentType     = UCHAR;
  // Set m_MachineByteOrder to the ByteOrder of the machine
  // Start out with file byte order == system byte order
  // this will be changed if we're reading a file to whatever
  // the file actually contains.
  if(ByteSwapper<int>::SystemIsBigEndian())
    m_MachineByteOrder = m_ByteOrder = BigEndian;
  else
    m_MachineByteOrder = m_ByteOrder = LittleEndian;

  // Set all values to a default value
  // Must check again -- memset!!!!

  //Analyze stuff
  //  memset sets the first n bytes in memory area s to the value of c
  //  (cothis->m_hdr.dime.dim[4]erted to an unsigned char).  It returns s.
  //  void *memset (void *s, int c, size_t n);
  memset(&(this->m_hdr),0, sizeof(struct dsr));

  //strcpy(this->m_hdr.hk.data_type,DataTypes[DT_INDEX_UNKNOWN]);
  /*Acceptable this->m_hdr.hk.data_type values are */
  /*"UNKNOWN","BINARY","CHAR","SHORT","INT","FLOAT","COMPLEX","DOUBLE","RGB" */
  this->m_hdr.hk.sizeof_hdr=sizeof(struct dsr);
  this->m_hdr.hk.db_name[0]='\0';
  this->m_hdr.hk.extents=16384;
  this->m_hdr.hk.session_error=0;
  this->m_hdr.hk.regular='r';
  this->m_hdr.hk.hkey_un0='\0';

  /*HeaderObj_dimension information*/
  this->m_hdr.dime.dim[0]=4;     //Usually 4 x,y,z,time
  this->m_hdr.dime.dim[1]=1;     //size_x;//number of columns
  this->m_hdr.dime.dim[2]=1;     //size_y;//number of rows
  this->m_hdr.dime.dim[3]=1;     //size_z;//number of slices
  this->m_hdr.dime.dim[4]=1;     //size_t;//number of volumes
  this->m_hdr.dime.dim[5]=1;
  this->m_hdr.dime.dim[6]=1;
  this->m_hdr.dime.dim[7]=1;

  /*labels voxel spatial unit */
  this->m_hdr.dime.vox_units[0]='\0';
  /*labels voxel calibration unit */
  this->m_hdr.dime.cal_units[0]='\0';

  this->m_hdr.dime.unused1=0;
  // Acceptable data values are DT_NONE, DT_UNKOWN, DT_BINARY,
  // DT_UNSIGNED_CHAR
  // DT_SIGNED_SHORT, DT_SIGNED_INT, DT_FLOAT, DT_COMPLEX, DT_DOUBLE,
  // DT_RGB, DT_ALL
  //this->m_hdr.dime.datatype=DataTypeKey[DT_INDEX_UNKNOWN];

  //this->m_hdr.dime.bitpix=DataTypeSizes[DT_INDEX_UNKNOWN];/*bits per pixel*/
  this->m_hdr.dime.dim_un0=0;

  //Set the voxel dimension fields:
  //A value of 0.0 for these fields implies that the value is unknown.
  //Change these values to what is appropriate for your data
  //or pass additional commathis->m_hdr.dime.dim[0] line arguments
  this->m_hdr.dime.pixdim[0]=0.0;//Unused field
  this->m_hdr.dime.pixdim[1]=1.0;//x_dimension
  this->m_hdr.dime.pixdim[2]=1.0;//y_dimension
  this->m_hdr.dime.pixdim[3]=1.0;//z_dimension
  this->m_hdr.dime.pixdim[4]=1.0;//t_dimension
  this->m_hdr.dime.pixdim[5]=1.0;
  this->m_hdr.dime.pixdim[6]=1.0;
  this->m_hdr.dime.pixdim[7]=1.0;
  // Assume zero offset in .img file, byte at which pixel data starts in
  // the HeaderObj file
  //byte offset in the HeaderObj file which voxels start
  this->m_hdr.dime.vox_offset=0.0;

  this->m_hdr.dime.roi_scale=0.0;
  this->m_hdr.dime.funused1=0.0;
  this->m_hdr.dime.funused2=0.0;
  this->m_hdr.dime.cal_max=0.0;  /*specify range of calibration values*/
  this->m_hdr.dime.cal_min=0.0;  /*specify range of calibration values*/
  this->m_hdr.dime.compressed=0; /*specify that the data file with extension .img is not compressed*/
  this->m_hdr.dime.verified=0;
  this->m_hdr.dime.glmax=0;      /*max value for all of the data set*/
  this->m_hdr.dime.glmin=0;      /*min value for all of the data set*/

  /*data_history*/
  this->m_hdr.hist.descrip[0]='\0';
  this->m_hdr.hist.aux_file[0]='\0';
  /*Acceptable values are*/
  /*0-transverse unflipped*/
  /*1-coronal unflipped*/
  /*2-sagittal unfipped*/
  /*3-transverse flipped*/
  /*4-coronal flipped*/
  /*5-sagittal flipped*/
  this->m_hdr.hist.orient=itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_RPI_TRANSVERSE; //default orientation is ITK_ANALYZE_TRANSVERSE

  this->m_hdr.hist.originator[0]='\0';
  this->m_hdr.hist.generated[0]='\0';
  this->m_hdr.hist.scannum[0]='\0';
  this->m_hdr.hist.patient_id[0]='\0';
  this->m_hdr.hist.exp_date[0]='\0';
  this->m_hdr.hist.exp_time[0]='\0';
  this->m_hdr.hist.hist_un0[0]='\0';
  this->m_hdr.hist.views=0;
  this->m_hdr.hist.vols_added=0;
  this->m_hdr.hist.start_field=0;
  this->m_hdr.hist.field_skip=0;
  this->m_hdr.hist.omax=0;
  this->m_hdr.hist.omin=0;
  this->m_hdr.hist.smax=0;
  this->m_hdr.hist.smin=0;
}

AnalyzeSliceImageIO::~AnalyzeSliceImageIO()
{
  //Purposefully left blank
}

void AnalyzeSliceImageIO::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

bool AnalyzeSliceImageIO::CanWriteFile(const char * FileNameToWrite)
{
  std::string filename(FileNameToWrite);
  // Data file name given?
  std::string::size_type imgPos = filename.rfind(".img");
  if ((imgPos != std::string::npos)
      && (imgPos == filename.length() - 4))
    {
    return true;
    }

  // Header file given?
  std::string::size_type hdrPos = filename.rfind(".hdr");
  if ((hdrPos != std::string::npos)
      && (hdrPos == filename.length() - 4))
    {
    return true;
    }

  // Compressed image given?
  std::string::size_type imggzPos = filename.rfind(".img.gz");
  if ((imggzPos != std::string::npos)
      && (imggzPos == filename.length() - 7))
    {
    return true;
    }
    
  return false;
}


//Set Data Type Values and min/max values
//////////////////////////////////////////////////////////////////////////
// Programmer: Hans J. Johnson
//       Date: 10/29/98
//   Function: DefineHeaderObjDataType
//  Algorithm: Set DataType Values appropriatly
// Func. Ret.:
//     Output:
//      Input: DataTypeIndex - Is one of the following
//              DT_INDEX_UNSIGNED_CHAR
//              DT_INDEX_SIGNED_SHORT   DT_INDEX_SIGNED_INT
//              DT_INDEX_FLOAT          DT_INDEX_DOUBLE
//              DT_INDEX_COMPLEX        DT_INDEX_RGB
//              DT_INDEX_BINARY         DT_INDEX_UNKNOWN
//////////////////////////////////////////////////////////////////////////
void  AnalyzeSliceImageIO::DefineHeaderObjectDataType()
{
  enum DataTypeIndex eNewType;
  switch(m_ComponentType)
    {
    case CHAR:
    case UCHAR:
      eNewType=ANALYZE_DT_INDEX_UNSIGNED_CHAR;
      break;
    case SHORT:
      eNewType=ANALYZE_DT_INDEX_SIGNED_SHORT;
      break;
    case USHORT:
      eNewType = SPMANALYZE_DT_INDEX_UNSIGNED_SHORT;
      break;
    case INT:
      eNewType=ANALYZE_DT_INDEX_SIGNED_INT;
      break;
    case UINT:
      eNewType=SPMANALYZE_DT_INDEX_UNSIGNED_INT;
      break;
    case FLOAT:
      eNewType=ANALYZE_DT_INDEX_FLOAT;
      break;
    case DOUBLE:
      eNewType=ANALYZE_DT_INDEX_DOUBLE;
      break;
      //case DATA_COMPLEX_FLOAT:
      //  eNewType=ANALYZE_DT_INDEX_COMPLEX;
      //  break;
      //case DATA_RGBTRIPLE:
      //  eNewType=ANALYZE_DT_INDEX_RGB;
      //  break;
      //case DATA_BINARY:
      //  eNewType=ANALYZE_DT_INDEX_BINARY;
      //  break;
      //  case
      //       DATA_UNKNOWN:
      //        eNewType=ANALYZE_DT_INDEX_UNKNOWN;
      //  break;
    default:
      eNewType=ANALYZE_DT_INDEX_UNKNOWN;
      ExceptionObject exception(__FILE__, __LINE__);
      exception.SetDescription("Pixel Type Unknown");
      throw exception;
    }
  m_hdr.dime.datatype=SCIDataTypeKey[eNewType];
  m_hdr.dime.bitpix=SCIDataTypeSizes[eNewType];
  strcpy(m_hdr.hk.data_type,SCIDataTypes[eNewType]);
  switch(m_hdr.dime.datatype)
    {
    case ANALYZE_DT_INDEX_BINARY:
      m_hdr.dime.glmax=1;  /*max value for all of the data set*/
      m_hdr.dime.glmin=0;  /*min value for all of the data set*/
      break;
    case ANALYZE_DT_INDEX_UNSIGNED_CHAR:
      m_hdr.dime.glmax=255;/*max value for all of the data set*/
      m_hdr.dime.glmin=0;  /*min value for all of the data set*/
      break;
    case ANALYZE_DT_INDEX_SIGNED_SHORT:
      //m_hdr.dime.glmax=0;/*max value for all of the data set*/
      //m_hdr.dime.glmin=0;/*min value for all of the data set*/
      break;
    case ANALYZE_DT_INDEX_FLOAT:
      //m_hdr.dime.glmax=0;/*max value for all of the data set*/
      //m_hdr.dime.glmin=0;/*min value for all of the data set*/
      break;
    case ANALYZE_DT_INDEX_DOUBLE:
      //m_hdr.dime.glmax=0;/*max value for all of the data set*/
      //m_hdr.dime.glmin=0;/*min value for all of the data set*/
      break;
    case ANALYZE_DT_INDEX_RGB:
      m_hdr.dime.glmax=255;/*max value for all of the data set*/
      m_hdr.dime.glmin=0;/*min value for all of the data set*/
      break;
    default:
      m_hdr.dime.glmax=0;  /*max value for all of the
                               data set*/
      m_hdr.dime.glmin=0;  /*min value for all of
                               the data set*/
      break;
    }
}

void AnalyzeSliceImageIO::Read(void* buffer)
{
  unsigned int dim;
  const unsigned int dimensions = this->GetNumberOfDimensions();
  unsigned int numberOfPixels = 1;
  for(dim=0; dim< dimensions; dim++ )
    {
    numberOfPixels *= m_Dimensions[ dim ];
    }

  char * const p = static_cast<char *>(buffer);
  //4 cases to handle
  //1: given .hdr and image is .img
  //2: given .img
  //3: given .img.gz
  //4: given .hdr and image is .img.gz
  //   Special processing needed for this case onl
  // NOT NEEDED const std::string fileExt = GetExtension(m_FileName);

  /* Returns proper name for cases 1,2,3 */
  std::string ImageFileName = GetImageFileName( m_FileName );
  //NOTE: gzFile operations act just like FILE * operations when the files
  // are not in gzip fromat.
  // This greatly simplifies the following code, and gzFile types are used
  // everywhere.
  // In addition, it has the added benifit of reading gzip compressed image
  // files that do not have a .gz ending.
  gzFile file_p = ::gzopen( ImageFileName.c_str(), "rb" );
  if( file_p == NULL )
    {
    /* Do a separate check to take care of case #4 */
    ImageFileName += ".gz";
    file_p = ::gzopen( ImageFileName.c_str(), "rb" );
    if( file_p == NULL )
      {
      ExceptionObject exception(__FILE__, __LINE__);
      std::string message="Analyze Data File can not be read: The following files were attempted:\n ";
      message += GetImageFileName( m_FileName );
      message += '\n';
      message += ImageFileName;
      message += '\n';
      exception.SetDescription(message.c_str());
      throw exception;
      }
    }

  // Seek through the file to the correct position, This is only necessary
  // when readin in sub-volumes
  // const long int total_offset = static_cast<long int>(tempX * tempY *
  //                                start_slice * m_dataSize)
  //    + static_cast<long int>(tempX * tempY * total_z * start_time *
  //          m_dataSize);
  // ::gzseek( file_p, total_offset, SEEK_SET );

  // read image in
  ::gzread( file_p, p, this->GetImageSizeInBytes());
  gzclose( file_p );
  SwapBytesIfNecessary( buffer, numberOfPixels );
}

itk::ImageIOBase::IOComponentType AnalyzeSliceImageIO::GetComponentType()
{
  return m_ComponentType;
}

std::string AnalyzeSliceImageIO::GetImageFile(const std::string& 
					      header_filename)
{
  return GetImageFileName(header_filename);
}

unsigned int AnalyzeSliceImageIO::GetNumberOfPixels(unsigned int axis)
{
  return m_Dimensions[axis];
}

double AnalyzeSliceImageIO::GetSpacing(unsigned int axis)
{
  return m_Spacing[axis];
}

double AnalyzeSliceImageIO::GetOrigin(unsigned int axis)
{
  return m_Origin[axis];
}

FILE* AnalyzeSliceImageIO::OpenImageFile(const std::string& filename)
{
  FILE* file_p = fopen( filename.c_str(), "rb" );
  return file_p;
}

void AnalyzeSliceImageIO::CloseImageFile(FILE* fp)
{
  if ( fp )
    fclose( fp );
}


// This method will only test if the header looks like an
// Analyze Header.  Some code is redundant with ReadImageInformation
// a StateMachine could provide a better implementation
bool AnalyzeSliceImageIO::CanReadFile( const char* FileNameToRead )
{
  std::string filename(FileNameToRead);

  // we check that the correction extension is given by the user
  std::string filenameext = GetExtension(filename);
  if(filenameext != std::string("hdr") 
    && filenameext != std::string("img.gz")
    && filenameext != std::string("img")
    )
    {
    return false;
    }

  const std::string HeaderFileName = GetHeaderFileName(filename);
  //
  // only try to read HDR files
  std::string ext = GetExtension(HeaderFileName);

  if(ext == std::string("gz"))
    {
    ext = GetExtension(GetRootName(HeaderFileName));
    }
  if(ext != std::string("hdr") && ext != std::string("img"))
    {
    return false;
    }

  std::ifstream   local_InputStream;
  local_InputStream.open( HeaderFileName.c_str(), 
                          std::ios::in | std::ios::binary );
  if( local_InputStream.fail() )
    {
    return false;
    }
  if( ! this->ReadBufferAsBinary( local_InputStream, 
                                  (void *)&(this->m_hdr), 
                                  sizeof(struct dsr) ) )
    {
    return false;
    }
  local_InputStream.close();

  // if the machine and file endianess are different
  // perform the byte swapping on it
  this->m_ByteOrder = this->CheckAnalyzeEndian(this->m_hdr);
  this->SwapHeaderBytesIfNecessary( &(this->m_hdr) );
#ifdef OMIT_THIS_CODE //It is OK for this flag to be set because the zlib will support the Unix compress files
  if(this->m_hdr.dime.compressed==1)
    {
    return false;
    //    ExceptionObject exception(__FILE__, __LINE__);
    //    exception.SetDescription("Unix compress file is not supported.");
    //    throw exception;
    }
#endif
  return true;
}

void AnalyzeSliceImageIO::ReadImageInformation()
{
  unsigned int dim;
  const std::string HeaderFileName = GetHeaderFileName( m_FileName );
  std::ifstream   local_InputStream;
  local_InputStream.open(HeaderFileName.c_str(),
                         std::ios::in | std::ios::binary);
  if( local_InputStream.fail())
    {
    ExceptionObject exception(__FILE__, __LINE__);
    exception.SetDescription("File cannot be read");
    throw exception;
    }
  if( ! this->ReadBufferAsBinary( local_InputStream, 
                                  (void *)&(this->m_hdr), 
                                  sizeof(struct dsr) ) )
    {
    ExceptionObject exception(__FILE__, __LINE__);
    exception.SetDescription("Unexpected end of file");
    throw exception;
    }
  local_InputStream.close();

  // if the machine and file endianess are different
  // perform the byte swapping on it
  this->m_ByteOrder=this->CheckAnalyzeEndian(this->m_hdr);
  if( this->m_MachineByteOrder != this->m_ByteOrder  )
    {
    this->SwapHeaderBytesIfNecessary( &(this->m_hdr) );
    }

  this->SetNumberOfDimensions(this->m_hdr.dime.dim[0]);
  switch( this->m_hdr.dime.datatype )
    {
    case ANALYZE_DT_BINARY:
      m_ComponentType = CHAR;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_UNSIGNED_CHAR:
      m_ComponentType = UCHAR;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_SIGNED_SHORT:
      m_ComponentType = SHORT;
      m_PixelType = SCALAR;
      break;
    case SPMANALYZE_DT_UNSIGNED_SHORT:
      m_ComponentType = USHORT;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_SIGNED_INT:
      m_ComponentType = INT;
      m_PixelType = SCALAR;
      break;
    case SPMANALYZE_DT_UNSIGNED_INT:
      m_ComponentType = UINT;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_FLOAT:
      m_ComponentType = FLOAT;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_DOUBLE:
      m_ComponentType = DOUBLE;
      m_PixelType = SCALAR;
      break;
    case ANALYZE_DT_RGB:
      // DEBUG -- Assuming this is a triple, not quad
      //image.setDataType( uiig::DATA_RGBQUAD );
      break;
    default:
      break;
    }
  //
  // set up the dimension stuff
  for(dim = 0; dim < this->GetNumberOfDimensions(); dim++)
    {
    this->SetDimensions(dim,this->m_hdr.dime.dim[dim+1]);
    this->SetSpacing(dim,this->m_hdr.dime.pixdim[dim+1]);
    }
  //
  // figure out re-orientation required if not in Coronal
  this->ComputeStrides();
  //Get Dictionary Information
  //Insert Orientation.
  {
//  char temp[348];
  //Important hk fields.
  itk::MetaDataDictionary &thisDic=this->GetMetaDataDictionary();
  std::string classname(this->GetNameOfClass());
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_InputFilterName, classname);

  itk::EncapsulateMetaData<std::string>(thisDic,ITK_ImageFileBaseName,std::string(this->m_hdr.hk.db_name,18));

  //Important dime fields
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_VoxelUnits,std::string(this->m_hdr.dime.vox_units,4));
  itk::EncapsulateMetaData<std::string>(thisDic,ANALYZESLICE_CALIBRATIONUNITS,std::string(this->m_hdr.dime.cal_units,8));
  itk::EncapsulateMetaData<short int>(thisDic,ITK_OnDiskBitPerPixel,this->m_hdr.dime.bitpix);
  itk::EncapsulateMetaData<float>(thisDic,SPM_ROI_SCALE,this->m_hdr.dime.roi_scale);
  itk::EncapsulateMetaData<float>(thisDic,ANALYZESLICE_CAL_MAX,this->m_hdr.dime.cal_max);
  itk::EncapsulateMetaData<float>(thisDic,ANALYZESLICE_CAL_MIN,this->m_hdr.dime.cal_min);
  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_GLMAX,this->m_hdr.dime.glmax);
  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_GLMIN,this->m_hdr.dime.glmin);

  for (dim=this->GetNumberOfDimensions(); dim>0; dim--)
  {
      if (m_hdr.dime.dim[dim] != 1)
          break;
  }
  itk::EncapsulateMetaData<int>(thisDic,ITK_NumberOfDimensions,dim);

  switch( this->m_hdr.dime.datatype)
    {
    case ANALYZE_DT_BINARY:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(char).name()));
      break;
    case ANALYZE_DT_UNSIGNED_CHAR:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned char).name()));
      break;
    case ANALYZE_DT_SIGNED_SHORT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(short).name()));
      break;
    case SPMANALYZE_DT_UNSIGNED_SHORT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned short).name()));
      break;
    case ANALYZE_DT_SIGNED_INT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(long).name()));
      break;
    case SPMANALYZE_DT_UNSIGNED_INT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned long).name()));
      break;
    case ANALYZE_DT_FLOAT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(float).name()));
      break;
    case ANALYZE_DT_DOUBLE:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(double).name()));
      break;
    case ANALYZE_DT_RGB:
      // DEBUG -- Assuming this is a triple, not quad
      //image.setDataType( uiig::DATA_RGBQUAD );
      break;
    default:
      break;
    }

  //Important hist fields
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_FileNotes,std::string(this->m_hdr.hist.descrip,80));
  itk::EncapsulateMetaData<std::string>(thisDic,ANALYZESLICE_AUX_FILE_NAME,std::string(this->m_hdr.hist.aux_file,24));

  {
    itk::AnalyzeSliceImageIO::ValidAnalyzeOrientationFlags temporient= static_cast<itk::AnalyzeSliceImageIO::ValidAnalyzeOrientationFlags>(this->m_hdr.hist.orient);
    //itk::EncapsulateMetaData<itk::AnalyzeSliceImageIO::ValidAnalyzeOrientationFlags>(thisDic,ITK_AnalyzeOrientation, temporient);
    itk::SpatialOrientation::ValidCoordinateOrientationFlags coord_orient;
    switch (temporient)
      {
      case itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_RPI_TRANSVERSE:
          coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
          break;
      case itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_PIR_SAGITTAL:
          coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR;
          break;
      case itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_RIP_CORONAL:
          // fall thru
      default:
          coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP;
      }
    itk::EncapsulateMetaData<itk::SpatialOrientation::ValidCoordinateOrientationFlags>(thisDic,ITK_CoordinateOrientation, coord_orient);
  }
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_FileOriginator,std::string(this->m_hdr.hist.originator,10));
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_OriginationDate,std::string(this->m_hdr.hist.generated,10));
  itk::EncapsulateMetaData<std::string>(thisDic,ANALYZESLICE_ScanNumber,std::string(this->m_hdr.hist.scannum,10));
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_PatientID,std::string(this->m_hdr.hist.patient_id,10));
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_ExperimentDate,std::string(this->m_hdr.hist.exp_date,10));
  itk::EncapsulateMetaData<std::string>(thisDic,ITK_ExperimentTime,std::string(this->m_hdr.hist.exp_date,10));

  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_O_MAX,this->m_hdr.hist.omax);
  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_O_MIN,this->m_hdr.hist.omin);
  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_S_MAX,this->m_hdr.hist.smax);
  itk::EncapsulateMetaData<int>(thisDic,ANALYZESLICE_S_MIN,this->m_hdr.hist.smin);
  }
  return;
}

/**
   *
   */
void
AnalyzeSliceImageIO
::WriteImageInformation(void)
{
  unsigned int dim;
  if(this->GetNumberOfComponents() > 1) 
    {
    ExceptionObject exception(__FILE__, __LINE__);
    std::string ErrorMessage=
      "More than one component per pixel not supported";
    exception.SetDescription(ErrorMessage.c_str());
    throw exception;
    }
  const std::string HeaderFileName = GetHeaderFileName( m_FileName );
  std::ofstream   local_OutputStream;
  local_OutputStream.open( HeaderFileName.c_str(),
                           std::ios::out | std::ios::binary );
  if( local_OutputStream.fail() )
    {
    ExceptionObject exception(__FILE__, __LINE__);
    std::string ErrorMessage="File cannot be written";
    ErrorMessage+=HeaderFileName;
    exception.SetDescription(ErrorMessage.c_str());
    throw exception;
    }
  {
  std::string temp;
  //Important hk fields.
  itk::MetaDataDictionary &thisDic=this->GetMetaDataDictionary();

  switch( this->m_hdr.dime.datatype)
    {
    case ANALYZE_DT_BINARY:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(char).name()));
      break;
    case ANALYZE_DT_UNSIGNED_CHAR:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned char).name()));
      break;
    case ANALYZE_DT_SIGNED_SHORT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(short).name()));
      break;
    case SPMANALYZE_DT_UNSIGNED_SHORT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned short).name()));
      break;
    case ANALYZE_DT_SIGNED_INT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(long).name()));
      break;
    case SPMANALYZE_DT_UNSIGNED_INT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(unsigned long).name()));
      break;
    case ANALYZE_DT_FLOAT:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(float).name()));
      break;
    case ANALYZE_DT_DOUBLE:
      itk::EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,std::string(typeid(double).name()));
      break;
    case ANALYZE_DT_RGB:
      // DEBUG -- Assuming this is a triple, not quad
      //image.setDataType( uiig::DATA_RGBQUAD );
      break;
    default:
      break;
    }

  itk::ExposeMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,temp);
  if (temp==std::string(typeid(char).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"BINARY",10);
  }
  else if (temp==std::string(typeid(unsigned char).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"CHAR",10);
  }
  else if (temp==std::string(typeid(short).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"SHORT",10);
  }
  else if (temp==std::string(typeid(unsigned short).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"USHORT",10);
  }
  else if (temp==std::string(typeid(long).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"INT",10);
  }
  else if (temp==std::string(typeid(unsigned long).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"UINT",10);
  }
  else if (temp==std::string(typeid(float).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"FLOAT",10);
  }
  else if (temp==std::string(typeid(double).name()))
  {
      strncpy(this->m_hdr.hk.data_type,"DOUBLE",10);
  }
  else
  {
      strncpy(this->m_hdr.hk.data_type,"UNKNOWN",10);
  }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,temp))
    {
    strncpy(this->m_hdr.hk.data_type,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_ImageFileBaseName,temp))
    {
    strncpy(this->m_hdr.hk.db_name,temp.c_str(),18);//Note this is necessary because the array is not necessarily null terminated.
    }
  //Important dime fields
  if(itk::ExposeMetaData<std::string>(thisDic,ITK_VoxelUnits,temp))
    {
    strncpy(this->m_hdr.dime.vox_units,temp.c_str(),4);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ANALYZESLICE_CALIBRATIONUNITS,temp))
    {
    strncpy(this->m_hdr.dime.cal_units,temp.c_str(),8);//Note this is necessary because the array is not necessarily null terminated.
    }

  itk::ExposeMetaData<short int>(thisDic,ITK_OnDiskBitPerPixel,this->m_hdr.dime.bitpix);
  itk::ExposeMetaData<float>(thisDic,SPM_ROI_SCALE,this->m_hdr.dime.roi_scale);
  itk::ExposeMetaData<float>(thisDic,ANALYZESLICE_CAL_MAX,this->m_hdr.dime.cal_max);
  itk::ExposeMetaData<float>(thisDic,ANALYZESLICE_CAL_MIN,this->m_hdr.dime.cal_min);
  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_GLMAX,this->m_hdr.dime.glmax);
  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_GLMIN,this->m_hdr.dime.glmin);
  //Important hist fields
  if(itk::ExposeMetaData<std::string>(thisDic,ITK_FileNotes,temp))
    {
    strncpy(this->m_hdr.hist.descrip,temp.c_str(),80);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ANALYZESLICE_AUX_FILE_NAME,temp))
    {
    strncpy(this->m_hdr.hist.aux_file,temp.c_str(),24);//Note this is necessary because the array is not necessarily null terminated.
    }

  {
    itk::SpatialOrientation::ValidCoordinateOrientationFlags coord_orient;
    if ( itk::ExposeMetaData<itk::SpatialOrientation::ValidCoordinateOrientationFlags>(thisDic,ITK_CoordinateOrientation, coord_orient) )
        {
        switch (coord_orient)
            {
        case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI:
            this->m_hdr.hist.orient=itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_RPI_TRANSVERSE;
            break;
        case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR:
            this->m_hdr.hist.orient=itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_PIR_SAGITTAL;
            break;
        case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP:
            this->m_hdr.hist.orient=itk::AnalyzeSliceImageIO::ITK_ANALYZE_ORIENTATION_RIP_CORONAL;
            break;
        default:
            break;
            }
        }
  }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_FileOriginator,temp))
    {
    strncpy(this->m_hdr.hist.originator,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_OriginationDate,temp))
    {
    strncpy(this->m_hdr.hist.generated,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ANALYZESLICE_ScanNumber,temp))
    {
    strncpy(this->m_hdr.hist.scannum,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_PatientID,temp))
    {
    strncpy(this->m_hdr.hist.patient_id,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_ExperimentDate,temp))
    {
    strncpy(this->m_hdr.hist.exp_date,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  if(itk::ExposeMetaData<std::string>(thisDic,ITK_ExperimentTime,temp))
    {
    strncpy(this->m_hdr.hist.exp_date,temp.c_str(),10);//Note this is necessary because the array is not necessarily null terminated.
    }

  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_O_MAX,this->m_hdr.hist.omax);
  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_O_MIN,this->m_hdr.hist.omin);
  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_S_MAX,this->m_hdr.hist.smax);
  itk::ExposeMetaData<int>(thisDic,ANALYZESLICE_S_MIN,this->m_hdr.hist.smin);
  }
  for( dim=0; dim< this->GetNumberOfDimensions(); dim++ )
    {
    //NOTE: Analyze dim[0] are the number of dims, and dim[1..7] are the actual dims.
    this->m_hdr.dime.dim[dim+1]  = m_Dimensions[ dim ];
    }
  //DEBUG--HACK It seems that analyze 7.5 requires 4 dimensions.
  this->m_hdr.dime.dim[0]= 4;
  for( dim=this->GetNumberOfDimensions();(int)dim < this->m_hdr.dime.dim[0];
       dim++ )
    {
    //NOTE: Analyze dim[0] are the number of dims, and dim[1..7] are the actual dims.
    this->m_hdr.dime.dim[dim+1]  = 1; //Hardcoded to be 1;
    }
  for( dim=0; dim< this->GetNumberOfDimensions(); dim++ )
    {
    //NOTE: Analyze pixdim[0] is ignored, and the number of dims are taken from dims[0], and pixdim[1..7] are the actual pixdims.
    this->m_hdr.dime.pixdim[dim+1]= m_Spacing[ dim ];
    }
  //The next funciton sets bitpix, and datatype, and data_type fields
  //Along with gl_min and gl_max feilds.
  this->DefineHeaderObjectDataType();

  local_OutputStream.write( (const char *)&(this->m_hdr), sizeof(struct dsr) );
  if( local_OutputStream.eof() )
    {
    ExceptionObject exception(__FILE__, __LINE__);
    exception.SetDescription("Unexpected end of file");
    throw exception;
    }
  local_OutputStream.close();
  return;
}


/**
   *
   */
void
AnalyzeSliceImageIO
::Write( const void* buffer)
{
  //Write the image Information before writing data
  this->WriteImageInformation();

  //NOTE: voidp is defined by zlib.h
  //NOTE: Need const_cast because voidp is "void*", so
  //      "const voidp" is "void* const", not "const void*".
  voidp p = const_cast<voidp>(buffer);
  const std::string ImageFileName = GetImageFileName( m_FileName );
  const std::string fileExt=GetExtension( m_FileName );
  // Check case where image is acually a compressed image
  if(!fileExt.compare( "gz" ))
    {
    // Open the *.img.gz file for writing.
    gzFile  file_p = ::gzopen( ImageFileName.c_str(), "wb" );
    if( file_p==NULL )
      {
      ExceptionObject exception(__FILE__, __LINE__);
      std::string ErrorMessage="Error, Can not write compressed image file for ";
      ErrorMessage+=m_FileName;
      exception.SetDescription(ErrorMessage.c_str());
      throw exception;
      }

#ifdef __OMIT_UNTIL_RGB_IS_NEEEDED
    if ( image.getDataType() == uiig::DATA_RGBTRIPLE )
      {
      // Analyze RGB images are stored in channels, where all the red components are stored
      // first, followed by the green and blue components for each plane of the volume.
      // This is stored in an image of RGBTRIPLE data structures, which are in memory
      // stored as (red,green,blue).  The triples need to be converted to channels for
      // each plane when writing out the image.

      // NOTE: Do NOT change this.  The math here is necessary for CImageStrided to
      // read files correctly
      for( register unsigned int l=0; l<tempT; l++ )
        {
        unsigned int volumeOffset = l*m_uiVolumeOffset;
        for( register unsigned int k=0; k<tempZ; k++ )
          {
          unsigned int planeVolOffset = k*m_uiPlaneOffset + volumeOffset;

          // Reading the red channel
          {
          for( register unsigned int j=0; j<tempY; j++ )
            {
            unsigned int rowOffset =    j*m_uiRowOffset;
            for ( register unsigned int i=0; i<tempX; i++ )
              {
              //NOTE: unsigned char * is used to do byte wise offsets The offsets are computed
              //in bytes
              ::gzwrite( file_p, &(static_cast<unsigned char *>(data)[(m_uiInitialOffset+planeVolOffset+rowOffset)*m_dataSize]) + (i*3), sizeof(unsigned char) );
              }
            }
          }

          // Reading the green channel
          {
          for( register unsigned int j=0; j<tempY; j++ )
            {
            unsigned int rowOffset =    j*m_uiRowOffset;
            for ( register unsigned int i=0; i<tempX; i++ )
              {
              //NOTE: unsigned char * is used to do byte wise offsets The offsets are computed
              //in bytes
              ::gzwrite( file_p, &(static_cast<unsigned char *>(data)[(m_uiInitialOffset+planeVolOffset+rowOffset)*m_dataSize]) + (i*3) + 1, sizeof(unsigned char) );
              }
            }
          }

          // Reading the blue channel
          {
          for( register unsigned int j=0; j<tempY; j++ )
            {
            unsigned int rowOffset =    j*m_uiRowOffset;
            for ( register unsigned int i=0; i<tempX; i++ )
              {
              //NOTE: unsigned char * is used to do byte wise offsets The offsets are computed
              //in bytes
              ::gzwrite( file_p, &(static_cast<unsigned char *>(data)[(m_uiInitialOffset+planeVolOffset+rowOffset)*m_dataSize]) + (i*3) + 2, sizeof(unsigned char) );
              }
            }
          }

          }
        }
      }
    else
#endif
      {
      ::gzwrite( file_p,p,this->GetImageSizeInBytes());
      }
    ::gzclose( file_p );
    //RemoveFile FileNameToRead.img so that it does not get confused with
    //FileNameToRead.img.gz
    //The following is a hack that can be used to remove ambiguity when an
    //uncompressed image is read, and then written as compressed.
    //This results in one *.hdr file being assosiated with a *.img and a
    // *.img.gz image file.
    //DEBUG -- Will this work under windows?
    std::string unusedbaseimgname= GetRootName(GetHeaderFileName(m_FileName));
    unusedbaseimgname+=".img";
    itksys::SystemTools::RemoveFile(unusedbaseimgname.c_str());
    }
  else
    {
    //No compression
    std::ofstream   local_OutputStream;
    local_OutputStream.open( ImageFileName.c_str(), std::ios::out | std::ios::binary );
    if( !local_OutputStream )
      {
      ExceptionObject exception(__FILE__, __LINE__);
      std::string ErrorMessage="Error opening image data file for writing.";
      ErrorMessage+=m_FileName;
      exception.SetDescription(ErrorMessage.c_str());
      throw exception;
      }
    local_OutputStream.write((const char *)p, this->GetImageSizeInBytes() );
    bool success = !local_OutputStream.bad();
    local_OutputStream.close();
    if( !success )
      {
      ExceptionObject exception(__FILE__, __LINE__);
      std::string ErrorMessage="Error writing image data.";
      ErrorMessage+=m_FileName;
      exception.SetDescription(ErrorMessage.c_str());
      throw exception;
      }
    //RemoveFile FileNameToRead.img.gz so that it does not get confused with FileNameToRead.img
    //The following is a hack that can be used to remove ambiguity when an
    //uncompressed image is read, and then written as compressed.
    //This results in one *.hdr file being assosiated with a *.img and a *.img.gz image file.
    //DEBUG -- Will this work under windows?
    std::string unusedbaseimgname= GetRootName(GetHeaderFileName(m_FileName));
    unusedbaseimgname+=".img.gz";
    itksys::SystemTools::RemoveFile(unusedbaseimgname.c_str());
    }
}
} // end namespace itk
