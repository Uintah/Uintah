all:timer.cpp timer.hpp
	cc -c timer.cpp -o timer.o -fPIC
	ar rcs libfunc_timer.a timer.o
	cp libfunc_timer.a lib/libfunc_timer.a
	cp timer.hpp lib/timer.hpp
clean:
	rm *.o
