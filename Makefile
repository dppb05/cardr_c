CC=gcc
CFLAGS=-Wall -c
EXEDIR=.

all: $(EXEDIR)/cardr

$(EXEDIR)/cardr: stex.o matrix.o util.o cardr.o
	$(CC) stex.o matrix.o util.o cardr.o -o $@ -lm

cardr.o: cardr.c
	$(CC) $(CFLAGS) cardr.c

util.o: util.c
	$(CC) $(CFLAGS) util.c

matrix.o: matrix.c
	$(CC) $(CFLAGS) matrix.c

stex.o: stex.c
	$(CC) $(CFLAGS) stex.c

clean:
	rm -rf *o $(EXEDIR)/cardr
