# include <sys/stat.h>
# include <stdio.h>
# include <unistd.h>

 int mk_dir_( char *argv)
 {
// Make a directory in linux /unix os using sys/stat.h and access it from fortran
// It will not work for windows (but who cares about windows anyway?)
 int i=mkdir(argv,0777);
// printf("%d", &i);
 return(i);
 }

 int rm_file_( char *argv) {
   int i=remove(argv);
   return(i);
 }
