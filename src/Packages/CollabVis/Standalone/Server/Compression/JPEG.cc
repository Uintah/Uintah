#include <jerror.h>
#include <string.h>
#include <Compression/JPEG.h>


namespace SemotusVisum {
namespace Compression {

const char * const
JPEGCompress::name = "JPEG";
const char * const
Compressor::name = "Compression";

/* ------------------ JPEG COMPRESSION ------------------------ */

/* Constructor. Create and initialize both the compression and
   decompression objects. */
JPEGCompress::JPEGCompress()
{
  init();
}

/* Constructor. Create and initialize both the compression and
   decompression objects. */
JPEGCompress::JPEGCompress(int quality)
{
  init();
  if (quality > 100) quality = 100;
  if (quality < 0)   quality = 0;
  jpeg_set_quality(&ccinfo, quality, false);
}


/* Deconstructor. Deallocate the compression and decompression objects. */
JPEGCompress::~JPEGCompress()
{
  jpeg_destroy_decompress(&dcinfo);
  jpeg_destroy_compress(&ccinfo);
}

//#define COPY_BUFFER
#define BAD_JPEG
/* Compression routine. */
int
JPEGCompress::compress(DATA * input,
		       int width,
		       int height,
		       DATA ** output,
		       int bps,
		       int delta)
{
  DATA * output_buffer;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;
  
#ifdef COPY_BUFFER
  int scanline;
  int image_delta = width % 4;
  int old_width = width;
  
  DATA * tmpbuf;

  if (image_delta != 0) image_delta = 4 - image_delta;
  
  width = width + image_delta;
  tmpbuf = scinew DATA[width * height * bps * sizeof(DATA)];
  for (scanline = 0; scanline < height; scanline++)
    {
      // Copy input to new buffer
      memcpy(&tmpbuf[scanline * width * bps * sizeof(DATA)],
	     &input[scanline * old_width * bps * sizeof(DATA)],
	     old_width * bps * sizeof(DATA));
      // Fill in pad info
      memset(&tmpbuf[scanline * old_width * bps * sizeof(DATA) + 1],
	     tmpbuf[scanline * old_width * bps * sizeof(DATA)],
	     image_delta);
    }
  
  
  input = tmpbuf;
#endif
  
  /* See if we need to allocate memory */
  if (*output == NULL)
    {
      *output = allocateMemory(width * height * bps); // The file shouldn't expand.
      if (*output == NULL)
	return -1;
    }
  output_buffer = *output;

  /* Do the compression */
  JSAMPROW row_pointer[1];

  int image_size = width * height;

  /* NOTE - DUE TO THE PADDING ISSUES, WE MAY LOSE THE LAST UP TO 3 PIXELS OF
     THE IMAGE. HOW DO WE FIX THIS? */
#ifdef BAD_JPEG
  ccinfo.image_width = width - width % 4;
  ccinfo.image_height = image_size / ccinfo.image_width;
#else
  ccinfo.image_width = width;
  ccinfo.image_height = height;
#endif
  
  jpeg_set_defaults(&ccinfo);

  // Speed up the process
#ifdef SGI
  ccinfo.dct_method = JDCT_FLOAT; // As the R10000 has a faster FPU than
                                  // integer processing unit.
#else
  ccinfo.dct_method = JDCT_FASTEST;
#endif
  
  /* Set data destination for compression */
  jpeg_buffer_dest(&ccinfo, (JSAMPLE *)output_buffer);

  /* Start the compressor */
  jpeg_start_compress(&ccinfo, TRUE);


  /* Process data */
  int row_stride = ccinfo.image_width * bps;
  
#ifndef BAD_JPEG
  row_pointer[0] = scinew JSAMPLE[row_stride + row_stride % 4];
#endif

#if 0 && (defined BAD_JPEG)
  ccinfo.image_width = width - width % 4;
  row_stride = ccinfo.image_width * bps;
#endif
  
  while (ccinfo.next_scanline < ccinfo.image_height)
    {
#ifdef BAD_JPEG
      row_pointer[0] = &input[ccinfo.next_scanline * row_stride];
#else
      memcpy(row_pointer[0], (void *)&input[ccinfo.next_scanline * row_stride],
	     row_stride);
#endif
      jpeg_write_scanlines(&ccinfo, row_pointer, 1);
    }
  
  /* Finish compression */
  jpeg_finish_compress(&ccinfo);

#ifdef COPY_BUFFER
  delete[] input;
#endif
  
  /* Return the # of bytes written */
  return ((dm_ptr)ccinfo.dest)->bytes_written;
}

/* Decompression routine. */
int
JPEGCompress::decompress(DATA * input,
			 int buffer_length,
			 DATA ** output,
			 int delta)
{
  DATA * output_start;
  DATA * output_buffer;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* Do the decompression */
  
  /* Specify data source for decompression */
  jpeg_buffer_src(&dcinfo, input, buffer_length);
  
  /* Read image info */
  jpeg_read_header(&dcinfo, TRUE);

  /* See if we need to allocate memory */
  if (*output == NULL)
    {
      *output = allocateMemory(dcinfo.output_width *
			       dcinfo.output_height *
			       dcinfo.output_components);
      if (*output == NULL)
	return -1;
    }
  output_buffer = *output;
  output_start = output_buffer;
  
  /* Start the decompression */
  jpeg_start_decompress(&dcinfo);

  /* Make a one-row-high sample array that will go away when done with image */
  int row_stride = dcinfo.output_width * dcinfo.output_components;
  int pad = row_stride % 4;
  JSAMPARRAY row_buffer;
  
  row_buffer = (*dcinfo.mem->alloc_sarray)
    ((j_common_ptr) &dcinfo, JPOOL_IMAGE, row_stride+pad, 1);
  
  /* Process data */
  while (dcinfo.output_scanline < dcinfo.output_height){
    memcpy(output_start, row_buffer[0], row_stride);
    output_start += row_stride + pad; 
  }

  int size =
    dcinfo.image_width * dcinfo.image_height * dcinfo.output_components;
  
  /* Finish the decompression */
  jpeg_finish_decompress(&dcinfo);
  
  /* Now output contains the uncompressed image. */
  return size;
}

/* Initializes data managers. Called from constructor. */
void
JPEGCompress::init()
{
  /* Init decompression */
  dcinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&dcinfo);
  dcinfo.src = NULL; // Temporarily lose this memory - reclaimed later.

  /* Init compression */
  ccinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&ccinfo);
  ccinfo.dest = NULL; // Temporarily lose this memory - reclaimed later.
  ccinfo.in_color_space = JCS_RGB;
  ccinfo.input_components = 3;
}

void
init_source(j_decompress_ptr cinfo)
{
  /* No-op */
}

boolean
fill_input_buffer(j_decompress_ptr cinfo)
{
  
  sm_ptr src = (sm_ptr) cinfo->src;

  src->pub.next_input_byte = src->buffer;
  src->pub.bytes_in_buffer = src->buffer_size;
    
  return TRUE;
}

void
skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
  sm_ptr src = (sm_ptr) cinfo->src;
  
  if (num_bytes > 0)
    {
      src->pub.next_input_byte += (size_t) num_bytes;
      src->pub.bytes_in_buffer -= (size_t) num_bytes;
    }
}

void
term_source(j_decompress_ptr cinfo)
{
  /* No-op */
}

void
jpeg_buffer_src(j_decompress_ptr cinfo, JSAMPLE * buffer,
			      int buffer_size)
{
  sm_ptr src;

  if (cinfo->src == NULL) {
    cinfo->src = (jpeg_source_mgr *) scinew sourceManager;

    src = (sm_ptr) cinfo->src;
    src->buffer = (JOCTET *)buffer;
  }

  src = (sm_ptr) cinfo->src;
  src->pub.init_source = init_source;
  src->pub.fill_input_buffer = fill_input_buffer;
  src->pub.skip_input_data = skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
  src->pub.term_source = term_source;
  src->pub.bytes_in_buffer = 0; /* forces fill_input_buffer on first read */
  src->pub.next_input_byte = NULL; /* until buffer loaded */
  
  src->buffer_size = buffer_size;

}

void 
init_destination (j_compress_ptr cinfo)
{
  int bufsize = cinfo->image_width*cinfo->image_height*3;
  dm_ptr dest = (dm_ptr) cinfo->dest;
  
  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = bufsize;
}

boolean 
empty_output_buffer(j_compress_ptr cinfo)
{
  /* This should never happen. If it does, we're hosed! */
  printf("eob\n");fflush(stdout);
  ERREXIT(cinfo, JERR_BUFFER_SIZE);
  return true;
  
}

void 
term_destination(j_compress_ptr cinfo)
{
  /* All data has been written. Now we just need to calculate the
     size of the compressed image. */
    
  int bytes_written = (cinfo->image_width * cinfo->image_height * 3) -
    cinfo->dest->free_in_buffer;
  
  dm_ptr dest = (dm_ptr) cinfo->dest;

  dest->bytes_written = bytes_written;
}

void 
jpeg_buffer_dest(j_compress_ptr cinfo, JSAMPLE * buffer)
{
  dm_ptr dest;
  
  /* Set up all our various options, allocate mem, etc. */
  if (cinfo->dest == NULL) /* first time for this JPEG object? */
    cinfo->dest = (jpeg_destination_mgr *) scinew destManager;

  dest = (dm_ptr) cinfo->dest;
  dest->pub.init_destination = init_destination;
  dest->pub.empty_output_buffer = empty_output_buffer;
  dest->pub.term_destination = term_destination;
  dest->buffer = buffer;
}

} // End namespace Compression
}
