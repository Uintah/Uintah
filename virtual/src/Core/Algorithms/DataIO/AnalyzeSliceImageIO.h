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
 *  AnalyzeSliceImageIO : Inherit from itkAnalyzeImageIO
 *               to allow for reading a slice at a time.
 *
 *  Written by:
 *   Darby Van Uitert
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2005
 *
 */


#ifndef __itkAnalyzeSliceImageIO_h
#define __itkAnalyzeSliceImageIO_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <fstream>
#include "itkImageIOBase.h"
#include "itkAnalyzeDbh.h"

#include <Core/Algorithms/DataIO/share.h>
namespace itk
{
/**
   * \ingroup IOFilters
   * \author Hans J. Johnson
   * \brief Class that defines how to read Analyze file format.
   * Analyze IMAGE FILE FORMAT - As much information as I can determine from the Medical image
   * formats web site, and the Analyze75.pdf file provided from the Mayo clinic.
   * A special note of thanks to Dennis P. Hanson (dph@mayo.edu) for his generous contributions
   * in getting this information correct.
   *
   * Analyze image file sets consist of at least 2 files:
   * REQUIRED:
   *    - an image file  ([basename].img or [basename].img.gz or [basename].img.Z)
   *          This contains the binary represenation of the raw voxel values.
   *          If the file is uncompressed, it should be of of size (sizeof(storagetype)*NX*NY*NZ(*NT).
   *          The format of the image file is very simple; containing usually
   *          uncompressed voxel data for the images in one of the several
   *          possible voxel formats:
   *             - 1 bit  packed binary (slices begin on byte boundaries)
   *             - 8 bit  (unsigned char) gray scale unless .lkup file present
   *             - 16 bit signed short
   *             - 32 bit signed integers or float
   *             - 24 bit RGB, 8 bits per channel
   *    - a header file  ([basename].hdr)
   *          This a 348 byte file 99.99% of all images that contains a binary represenation of the C-struct
   *          defined in this file.  The analyze 7.5 header structure may, however, be extended beyond this minimal defintion
   *          to encompase site specific information, and would have more than 348 bytes.  Given that the
   *          ability to extend the header has rarely been used, this implementation of the Analyze 7.5
   *          file will only read the first 348 bytes into the structure defined in this file, and all informaiton beyond the
   *          348 bytes will be ignored.
   * OPTIONAL:
   *    - a color lookup file ([basename].lkup)
   *      The .lkup file is a plain ASCII text file that contains 3 integer values between 0 and 255
   *      on each line.  Each line of the lkup file represents one color table entry for the Red,
   *      Green and Blue color components, respectively.  The total dynamic range of the image
   *      is divided by the number of colors in color table to determine mapping of the image through
   *      these colors.
   *       For example, an 8-color 'rainbow colors' lookup table is represented as:
   *       ===========================
   *       255 0 0
   *       255 128 0
   *       255 255 0
   *       128 255 0
   *       0 255 0
   *       0 0 255
   *       128 0 255
   *       255 0 255
   *       ===========================
   *    - an object file ([basename].obj)
   *      A specially formated file with a mapping between object name and image code used to associate
   *      image voxel locations with a label.  This file is run length encoded to save disk storage.
   */
class SCISHARE AnalyzeSliceImageIO : public ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef AnalyzeSliceImageIO            Self;
  typedef ImageIOBase  Superclass;
  typedef SmartPointer<Self>  Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AnalyzeSliceImageIO, Superclass);

  /*-------- This part of the interfaces deals with reading data. ----- */

  /** Determine if the file can be read with this ImageIO implementation.
       * \author Hans J Johnson
       * \param FileNameToRead The name of the file to test for reading.
       * \post Sets classes ImageIOBase::m_FileName variable to be FileNameToWrite
       * \return Returns true if this ImageIO can read the file specified.
       */
  virtual bool CanReadFile(const char* FileNameToRead) ;

  /** Set the spacing and dimension information for the set filename. */
  virtual void ReadImageInformation();

  /** Reads the data from disk into the memory buffer provided. */
  virtual void Read(void* buffer);

  itk::ImageIOBase::IOComponentType GetComponentType();

  std::string GetImageFile(const std::string& header_filename);

  unsigned int GetNumberOfPixels(unsigned int axis);

  double GetSpacing(unsigned int axis);

  double GetOrigin(unsigned int axis);

  FILE* OpenImageFile(const std::string& filename);

  void CloseImageFile(FILE*);

  /*-------- This part of the interfaces deals with writing data. ----- */

  /** Determine if the file can be written with this ImageIO implementation.
       * \param FileNameToWrite The name of the file to test for writing.
       * \author Hans J. Johnson
       * \post Sets classes ImageIOBase::m_FileName variable to be FileNameToWrite
       * \return Returns true if this ImageIO can write the file specified.
       */
  virtual bool CanWriteFile(const char * FileNameToWrite);

  /** Set the spacing and dimension information for the set filename. */
  virtual void WriteImageInformation();

  /** Writes the data to disk from the memory buffer provided. Make sure
       * that the IORegions has been set properly. */
  virtual void Write(const void* buffer);


protected:
  AnalyzeSliceImageIO();
  ~AnalyzeSliceImageIO();
  void PrintSelf(std::ostream& os, Indent indent) const;
private:
  /**
       * \enum ValidAnalyzeOrientationFlags
       * Valid Orientation values for objects
       * - Key  Description           Origin   dims[1]  dims[2]  dims[3]
       * - =================================================================
       * - 0    transverse-unflipped   IRP       R->L     P->A    I->S
       * - 1    coronal-unflipped      IRP       R->L     I->S    P->A
       * - 2    sagittal-unflipped     IRP       P->A     I->S    R->L
       * - 3    transverse-flipped     IRA       R->L     A->P    I->S
       * - 4    coronal-flipped        SRP       R->L     S->I    P->A
       * - 5    sagittal-flipped       ILP       P->A     I->S    L->R
       * - Where the Origin disignators are with respect to the patient
       * - [(I)nferior|(S)uperior] [(L}eft|(R)ight] [(A)nterior|(P)osterior]
       * \note Key's 0-5 correspond to the Analyze v7.5 orientations, and should not be changed.
       */
  typedef enum {
    ITK_ANALYZE_ORIENTATION_RPI_TRANSVERSE=0,        /**< Denotes a transverse data orientation Right-->Left, */
    ITK_ANALYZE_ORIENTATION_RIP_CORONAL   =1,        /**< Denotes a coronal data orientation */
    ITK_ANALYZE_ORIENTATION_PIR_SAGITTAL  =2,        /**< Denotes a sagittal data orientation */
    ITK_ANALYZE_ORIENTATION_RAI_TRANSVERSE_FLIPPED=3,/**<  */
    ITK_ANALYZE_ORIENTATION_RSP_CORONAL_FLIPPED=4,   /**<  */
    ITK_ANALYZE_ORIENTATION_PIL_SAGITTAL_FLIPPED=5   /**<  */
  } ValidAnalyzeOrientationFlags;


  AnalyzeSliceImageIO(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  void SwapBytesIfNecessary(void * buffer, unsigned long numberOfPixels);
  /**
       * \author Hans J. Johnson
       * Performs byte swapping of the Analyze Image header if necessary.
       * \param imageheader An Analyze 7.5 compliant image header.
       * \return void
       */
  void SwapHeaderBytesIfNecessary( struct dsr * const imageheader );

  /**
       * \author Hans J. Johnson
       * Defines the header object data type feilds according to Analyze v7.5 specifications
       * \return nothing
       */
  void  DefineHeaderObjectDataType(void);
#if defined(REORIENT_IMAGES)
  void ReorientIfNecessary(char *p);
  struct ipl_dimensions {
    unsigned int slicestride;
    unsigned int rowstride;
    unsigned int componentstride;
    unsigned int pixelsize;
    //
    // xsize,ysize,zsize == size in each direction in pixesls
    unsigned int xsize;
    unsigned int ysize;
    unsigned int zsize;
  };
  /**
       * \author Kent Williams
       * Get values needed to re-orient image data to
       * Coronal scan order
       * \param dim - structure to fill in
       * \return nothing
       */
  void GetAllDimensions(ipl_dimensions &dim);
  ipl_dimensions m_old_dim,m_new_dim;
#endif
  /**
       * \author Hans J. Johnson
       * Check the endedness of the header file.
       * \param temphdr - a reference to the header structure
       * \return The endedness of the file
       */
  ImageIOBase::ByteOrder CheckAnalyzeEndian(const struct dsr &temphdr);
  /**  All of the information read in from the header file */
  struct dsr m_hdr;
  ImageIOBase::ByteOrder m_MachineByteOrder;
};
extern const char *const ANALYZESLICE_ScanNumber;
extern const char *const ANALYZESLICE_O_MAX;
extern const char *const ANALYZESLICE_O_MIN;
extern const char *const ANALYZESLICE_S_MAX;
extern const char *const ANALYZESLICE_S_MIN;
extern const char *const ANALYZESLICE_CAL_MAX;
extern const char *const ANALYZESLICE_CAL_MIN;
extern const char *const ANALYZESLICE_GLMAX;
extern const char *const ANALYZESLICE_GLMIN;
extern const char *const ANALYZESLICE_AUX_FILE_NAME;
extern const char *const ANALYZESLICE_CALIBRATIONUNITS;

} // end namespace itk

#endif // __itkAnalyzeSliceImageIO_h
