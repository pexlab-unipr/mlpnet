# Name of the executable
TARGET ?= main_peaks

# Directory where DeTecTUM is located
DETECTUM_DIR ?= ../../detectum

# Addition compiler flags (e.g. EXTRA_CFLAGS="-O3 -fopt-info-vec=vec.log")
EXTRA_CFLAGS ?=

CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic $(EXTRA_CFLAGS)
LFLAGS = -lm

SRC = $(DETECTUM_DIR)/src/detectum.c src/mlpnet.c demo/$(TARGET).c
INC = -I inc -I $(DETECTUM_DIR)/inc

.PHONY: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(INC) $(SRC) $(LFLAGS) -o $@
