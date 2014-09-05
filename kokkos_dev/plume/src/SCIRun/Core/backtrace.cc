// backtrace.c
// Copyright 2005 McKay Davis



#if defined(__GNUC__) && defined(__linux)
#include <malloc.h>
#include <stdio.h>
#include <bfd.h>
#include <execinfo.h> // backtrace and backtrace_symbols
#define __USE_GNU
#include <dlfcn.h>    // dlinfo  and dladdr

extern "C" {
  typedef struct {
    asymbol **symbol_table;
    unsigned long address;
    unsigned int line;
    const char *file;
    const char *function;
    int valid;
  } symbol_info_t;
  
  
  extern char * cplus_demangle (const char *, int);
  
  static void
  symbol_info_callback(bfd *binary_file_descriptor,
		       asection *bfd_section,
		       void *infoptr)
  {
    unsigned long offset = 0;
    symbol_info_t *info = (symbol_info_t *)infoptr;
    if (!info || info->valid) return;
    
    offset = info->address - bfd_section->vma;
    //bfd_get_section_vma(binary_file_descriptor,bfd_section);
    offset -= offset % sizeof (void *);
    if (offset >= 0 && offset <= bfd_get_section_size/*_before_reloc*/(bfd_section))
      info->valid = bfd_find_nearest_line(binary_file_descriptor, bfd_section, 
					  info->symbol_table, offset,
					  &info->file, &info->function, 
					  &info->line);
  }
   
  
  void
  backtrace_linux( int stack_depth, void *stack_addresses[] )
  {
    //  const int stack_depth = backtrace(stack_addresses, 1024);
    const char *soextension = ".so";
    const char *indentation = "           ";
    char **names = backtrace_symbols(stack_addresses, stack_depth);
    char *demangled = 0;
    symbol_info_t info;
    unsigned int i = 0, j = 0, k = 0;
    bfd *binary_file_descriptor;
    Dl_info dlinfo;
    
    FILE *file = stderr;
    
    if (!stack_depth) {
      fprintf(file, "Backtrace not available!\n");
      return;
    }
    
    bfd_init();
    bfd_set_default_target ("i386");
    
    fprintf(file, "Backtrace:\n");
    // Note: [yarden] skipp the first two enteries as they refer to CCAException internals.
    for (i = 2; i < (unsigned int)stack_depth; i++)
      {
	if (!dladdr(stack_addresses[i], &dlinfo)) {
	  fprintf (file,"%sAddress cannot be translated to symbol: %s\n",
		   indentation, names[i]);
	  continue;
	}
	
	if (!(binary_file_descriptor = bfd_openr(dlinfo.dli_fname, 0)))
	  continue;
	
	if (!bfd_check_format(binary_file_descriptor, bfd_object)) {
	  bfd_close(binary_file_descriptor);
	  continue;
	}
	
	if (!bfd_read_minisymbols(binary_file_descriptor, 0, 
				  (void **)&info.symbol_table, &j))
	  bfd_read_minisymbols(binary_file_descriptor, 1, 
			       (void **)&info.symbol_table, &j);
	
	if (!info.symbol_table) {
	  fprintf(file, "%s,Cannot load symbol table for symbol %s in %s\n",
		  indentation, dlinfo.dli_sname, dlinfo.dli_fname);
	  bfd_close(binary_file_descriptor);
	  continue;
	}
	
	info.address = (unsigned long)stack_addresses[i];
	j = k = 0;
	while (dlinfo.dli_fname[j])
	  if (dlinfo.dli_fname[j++] != soextension[k++])
	    k = 0;
	if (k == 3)
	  info.address -= (unsigned long)dlinfo.dli_fbase;
	
	info.valid = 0;
	info.function = 0;
	bfd_map_over_sections(binary_file_descriptor,symbol_info_callback,&info);
	free(info.symbol_table);
	
	if (!info.function)
	  info.function = dlinfo.dli_sname;
	if (info.function && (demangled = cplus_demangle(info.function, 3)))
	  info.function = demangled;
	
	if (!info.valid) {
	  fprintf(file, 
		  "%sCannot find line information for symbol %s in %s: %s\n"
		  "%s%sfname: %s fbase: %p  sname: %s saddr: %p"
		  "  addr: %p info.address: 0x%lx\n",
		  indentation, info.function, dlinfo.dli_fname, names[i], indentation,
		  indentation, dlinfo.dli_fname,dlinfo.dli_fbase, dlinfo.dli_sname,
		  dlinfo.dli_saddr, stack_addresses[i], info.address);
	} else {
	  fprintf(file, "Frame %d: %s\n%s\t", i-2, info.function,indentation);
	  if (info.file)
	    fprintf(file,"@[%s,  line %d]\n", info.file, info.line);
	  else if(dlinfo.dli_fname)
	    fprintf(file,"@[%s]\n", dlinfo.dli_fname);
	  else
	    fprintf(file,"@[unknown file]\n");
	}
	
	if (demangled) {
	  free (demangled);
	  demangled = 0;
	}
	
	bfd_close(binary_file_descriptor);
      }
    free(names);
    fflush(file);
  }
}

#endif
