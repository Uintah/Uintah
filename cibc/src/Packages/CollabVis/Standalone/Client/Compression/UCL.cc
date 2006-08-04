#include <Compression/UCL.h>


namespace SemotusVisum {
namespace Compression {
/* ------------------ UCL COMPRESSION ------------------------ */

/* Constructor. Initialize the UCL library. */
UCLCompress::UCLCompress() : compressionLevel(1)
{
  if (ucl_init() != UCL_E_OK)
    printf("ucl_init() failed !!!\n");
}

/* Constructor. Initialize the UCL library. */
UCLCompress::UCLCompress(int level) : compressionLevel(level)
{
  // Bound the compression level.
  if (compressionLevel > 10)
    compressionLevel = 10;
  if (compressionLevel < 1)
    compressionLevel = 1;
  
  if (ucl_init() != UCL_E_OK)
    printf("ucl_init() failed !!!\n");
}

/* Deconstructor. No operations necessary */
UCLCompress::~UCLCompress()
{
}

/* Compression routine. */
int
UCLCompress::compress(DATA * input,
		      int width,
		      int height,
		      DATA ** output,
		      int delta)
{
  ucl_byte *out;
  ucl_uint in_len;
  ucl_uint out_len;
  int ret;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory */
  in_len = width*height*sizeof(DATA)*3;
  if (*output == NULL)
    {
      *output = allocateMemory(in_len + (in_len / 8) + 256);
      if (*output == NULL)
	return -1;
    }
  out = *output;

  /* Do the compression */
  ret = ucl_nrv2b_99_compress(input,in_len,
			      out,&out_len,
			      NULL,
			      compressionLevel,
			      NULL,
			      NULL);
  if (ret != UCL_E_OK)
    return -ret;

  return out_len;
}

/* Decompression routine. */
int
UCLCompress::decompress(DATA * input,
			int buffer_length,
			DATA ** output,
			int delta)
{
  ucl_byte *out;
  ucl_uint out_len;
  int ret;
  
  /* Do some sanity checking */
  if (!input || !output)
    return -1;

  /* See if we need to allocate memory.
     NOTE - We don't know how much memory to allocate! Argh! */
  if (*output == NULL)
    {
#if 0
      *output = allocateMemory(buffer_length);
      if (*output == NULL)
	return -1;
#else
      return -1; // Since we don't know how to find the new size of the
                 // data.
#endif
	
    }
  out = *output;

  /* Do the decompression */
  ret = ucl_nrv2b_decompress_8(input,buffer_length,out,&out_len,NULL);
  if (ret != UCL_E_OK)
    return -ret;

  return out_len;
}

} // End namespace Compression
}
