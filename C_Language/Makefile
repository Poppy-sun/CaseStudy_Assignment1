CC = mpicc
CFLAGS = -I/opt/homebrew/opt/lapack/include -Wall
LDFLAGS = -L/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas -lm

TARGET = TSQR_Q3
SECOND_TARGET = TSQR_Q2

SOURCES = tsqr_scaling_test.c parallel_TSQR.c
SECOND_SOURCES = TSQR_Q2.c parallel_TSQR.c
HEADERS = parallel_TSQR.h
OBJECTS = $(SOURCES:.c=.o)
SECOND_OBJECTS = $(SECOND_SOURCES:.c=.o)

all: $(TARGET) $(SECOND_TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(SECOND_TARGET): $(SECOND_OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(SECOND_OBJECTS) $(TARGET) $(SECOND_TARGET) scaling_results.csv *.png 

run_Q3: $(TARGET)
	mpirun -np 4 ./$(TARGET)

run_Q2: $(SECOND_TARGET)
	mpirun -np 4 ./$(SECOND_TARGET)
 