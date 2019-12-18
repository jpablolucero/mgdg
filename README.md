# Compile-time capable high order multigrid DG solver

After a fresh clone of this repository, the compilation will not lead to a compile-time calculation, but to a series of problem solutions in different size meshes.

To execute an example compile-time calculation, replace the line
```
const volatile auto x1  = make_main<128,1>();
```
for
```
const volatile auto x1  = mymain<4,1>();
```
at ```main()```, and the line
```
const auto solution = richardson<p,false,mf>(A,rhs,1.E-8) ;
```
for
```
const auto solution = richardson<p,false,mf,false>(A,rhs,1.E-8) ;
```
at ```mymain()```.

Edit the ```Makefile``` to include the location of your installation of ```gcc``` version 8.2 or newer and run ```make static```.

The compilation should take a couple minutes and occupy around 4 GB memory.

A file ```mgrid.s``` will be created with the assembly result of a compile-time executed calculation.

TODO: Documentation.
