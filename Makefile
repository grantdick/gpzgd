AR:=ar
CC:=clang
SRCDIR:=src
OBJDIR:=build
BINDIR:=dist

INCS:=$(wildcard $(SRCDIR)/*.h)

MAKEFLAGS:=-j
CFLAGS:=-std=gnu99 -Wall -Wextra -pedantic -march=native -O3 -fopenmp -g
# CFLAGS:=-std=gnu99 -Wall -Wextra -pedantic -march=native -O3 -g
IFLAGS:=
LFLAGS:=-lm

UTIL_OBJS:=$(OBJDIR)/data_set.o $(OBJDIR)/readline.o $(OBJDIR)/rng.o
ALG_OBJS:=$(OBJDIR)/gp.o
MAIN_OBJS:=$(OBJDIR)/main.o $(OBJDIR)/cmd_args.o $(OBJDIR)/measurement.o $(OBJDIR)/problem.o

BIN:=$(BINDIR)/regression

all: $(BIN)

$(BINDIR)/regression: $(MAIN_OBJS) $(ALG_OBJS) $(UTIL_OBJS)
	@echo linking $@ from $^
	@mkdir -p $(BINDIR)
	@$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c $(INCS)
	@echo compiling $< into $@
	@mkdir -p $(OBJDIR)
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)

nuke: clean
	@rm -rf $(INCDIR) $(BINDIR)

strip: all
	@echo running strip on $(BIN)
	@strip $(BIN)
