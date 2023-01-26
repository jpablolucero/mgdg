# Compile-time capable high order multigrid DG solver

After a fresh clone of this repository, the compilation will not lead to a compile-time calculation, but to a series of problem solutions in different size meshes.

Make sure GCC 10 is installed and the GNU EIGEN library. Provide the locations of those libraries in the Makefile.

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

To obtain the solution we need to disassemble the binary (note that we don't need the source code) as follows:
```
readelf -x .data mgrid

Hex dump of section '.data':
  0x00404040 00000000 00000000 00000000 00000000 ................
  0x00404050 00000000 00000000 00000000 00000000 ................
  0x00404060 df50b595 9696863f da992585 8787b73f .P.....?..%....?
  0x00404070 79cff075 7878b83f 3ab6bdfc ffffbf3f y..uxx.?:......?
  0x00404080 3ab6bdfc ffffbf3f 79cff075 7878b83f :......?y..uxx.?
  0x00404090 da992585 8787b73f df50b595 9696863f ..%....?.P.....?
```
This is the solution to a problem with 4 linear elements, so a vector of 8 doubles.

The values are stored reversed by pairs so for example, the first value `df50b595 9696863f` should be read `3f 86 96 96 95 b5 50 df` which converted to double is `0.011029411739095383`.

TODO: Documentation.
