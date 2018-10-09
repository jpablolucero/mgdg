G++ = /home/lucerolo/workspace/gcc/install/bin/g++

mgrid: mgrid.cc
	@$(G++) -I/home/lucerolo/workspace/blaze/install/include -march=native -std=c++1z -Ofast -DNDEBUG -ftree-vectorizer-verbose=9 -fopt-info-optall=vec_report -Wa,-adhln -g mgrid.cc -o mgrid > mgrid.s

clean:
	@rm -rf mgrid vec_report mgrid.s *~ \#*

run: mgrid
	@ulimit -s unlimited
	@./mgrid
