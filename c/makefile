# Name of the main file
MAIN ?= main_peaks

# Directory where DeTecTUM is located
DETECTUM_DIR ?= ../../detectum

# Addition compiler options (e.g. OPTIONS="-O3 -fopt-info-vec=vec.log")
OPTIONS ?=

COMPILER = gcc
CFLAGS = -Wall -Wextra $(OPTIONS)
LFLAGS = -lm

TARGET = $(MAIN)
SRC = src/mlpnet.c demo/$(MAIN).c $(DETECTUM_DIR)/src/detectum.c
INC = -I inc -I $(DETECTUM_DIR)/inc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(CFLAGS) $(INC) -o $@ $(SRC) $(LFLAGS)

clean:
	rm -f $(TARGET)
