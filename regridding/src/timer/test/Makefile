TARGET=timer_test
CC=g++
#CC=/usr/local/bin/g++
CFLAGS=-Wall -g -pipe -DFUNC_TIMER_ENABLED
CPPFLAGS=-I../lib
LDFLAGS=../timer.o

SOURCES=$(wildcard *.cpp)
OBJECTS=$(patsubst %.cpp,%.o,$(SOURCES))
HEADERS=$(wildcard include/*.h)

.PHONY:all
all: $(TARGET)

.PHONY:clean
clean:
	rm -f *.o
%.o:%.cpp $(HEADERS)
	$(CC) -o $@ -c $(CPPFLAGS) $(CFLAGS) $<
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $(LDFLAGS) $^
