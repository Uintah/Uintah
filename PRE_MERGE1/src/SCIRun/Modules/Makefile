dirs = Image 

all::
	@ for i in ${dirs};               \
          do                              \
		  echo "";                \
		  echo "Making $$i";      \
		  (cd $$i && make);       \
	  done

quick::
	@ for i in ${dirs};               \
          do                              \
		  echo "";                \
		  echo "Making $$i";      \
		  (cd $$i && make quick); \
	  done

clean::
	@ for i in ${dirs};                \
          do                               \
		  echo "";                 \
		  echo "Cleaning $$i";     \
		  (cd $$i && make clean);  \
	  done

soclean::
	@ for i in ${dirs};                 \
          do                                \
		  echo "";                  \
		  echo ".so Cleaning $$i";  \
		  (cd $$i && make soclean); \
	  done
